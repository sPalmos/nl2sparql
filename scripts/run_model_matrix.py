"""Run the model matrix: ontologies x models, one repetition each.

For every ontology it activates the package once, then runs each model in turn.
Results are appended to output/results_model_matrix_runs.csv immediately after each run, so an
interrupted sweep loses at most the run in flight. A final matrix
(output/results_model_matrix.csv) is written at the end and also printed.

Usage:
    python scripts/run_model_matrix.py                     # everything
    python scripts/run_model_matrix.py --only bestiary spider-world-1
    python scripts/run_model_matrix.py --models qwen2.5-coder:32b
    python scripts/run_model_matrix.py --dry-run           # print commands, run nothing
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES_DIR = ROOT / "ontology_packages"
DATASETS_DIR = ROOT / "evaluation" / "datasets"

RUNS_CSV = ROOT / "output" / "results_model_matrix_runs.csv"
MATRIX_CSV = ROOT / "output" / "results_model_matrix.csv"

MODELS = [
    # previous generation (2024) - kept as the reference point
    "qwen2.5-coder:7b",
    "qwen2.5-coder:32b",
    "llama3.1:8b",
    "deepseek-r1:32b",
    # current generation
    "qwen3:8b",
    "qwen3-coder:30b",
]

# ontology key -> (package dir prefix, dataset filename)
# The script checks that each package and dataset exists before running and
# reports any it cannot find.
ONTOLOGIES = {
    "bestiary":                           ("bestiary",                           "bestiary_sandro_eval_dataset.json"),
    "ck25":                               ("ck25",                               "ck25_text2sparql_combined_eval_dataset.json"),
    "spider-battle-death":                ("spider-battle-death",                "spider4sparql_sandro/spider_battle_death_combined_eval_dataset.json"),
    "spider-car-1":                       ("spider-car-1",                       "spider4sparql_sandro/spider_car_1_combined_eval_dataset.json"),
    "spider-concert-singer":              ("spider-concert-singer",              "spider4sparql_sandro/spider_concert_singer_combined_eval_dataset.json"),
    "spider-course-teach":                ("spider-course-teach",                "spider4sparql_sandro/spider_course_teach_combined_eval_dataset.json"),
    "spider-cre-doc-template-mgt":        ("spider-cre-doc-template-mgt",        "spider4sparql_sandro/spider_cre_doc_template_mgt_combined_eval_dataset.json"),
    "spider-dog-kennels":                 ("spider-dog-kennels",                 "spider4sparql_sandro/spider_dog_kennels_combined_eval_dataset.json"),
    "spider-employee-hire-evaluation":    ("spider-employee-hire-evaluation",    "spider4sparql_sandro/spider_employee_hire_evaluation_combined_eval_dataset.json"),
    "spider-flight-2":                    ("spider-flight-2",                    "spider4sparql_sandro/spider_flight_2_combined_eval_dataset.json"),
    "spider-museum-visit":                ("spider-museum-visit",                "spider4sparql_sandro/spider_museum_visit_combined_eval_dataset.json"),
    "spider-network-1":                   ("spider-network-1",                   "spider4sparql_sandro/spider_network_1_combined_eval_dataset.json"),
    "spider-pets-1":                      ("spider-pets-1",                      "spider4sparql_sandro/spider_pets_1_combined_eval_dataset.json"),
    "spider-poker-player":                ("spider-poker-player",                "spider4sparql_sandro/spider_poker_player_combined_eval_dataset.json"),
    "spider-real-estate-properties":      ("spider-real-estate-properties",      "spider4sparql_sandro/spider_real_estate_properties_combined_eval_dataset.json"),
    "spider-singer":                      ("spider-singer",                      "spider4sparql_sandro/spider_singer_combined_eval_dataset.json"),
    "spider-student-transcripts-tracking":("spider-student-transcripts-tracking","spider4sparql_sandro/spider_student_transcripts_tracking_combined_eval_dataset.json"),
    "spider-tvshow":                      ("spider-tvshow",                      "spider4sparql_sandro/spider_tvshow_combined_eval_dataset.json"),
    "spider-voter-1":                     ("spider-voter-1",                     "spider4sparql_sandro/spider_voter_1_combined_eval_dataset.json"),
    "spider-world-1":                     ("spider-world-1",                     "spider4sparql_sandro/spider_world_1_combined_eval_dataset.json"),
}

CSV_FIELDS = [
    "ontology", "model", "status", "questions", "scored",
    "accuracy", "macro_f1", "macro_precision", "macro_recall",
    "svr_initial", "svr_final", "execution_success",
    "correction_rate", "avg_iterations", "avg_latency_ms",
    "wall_seconds", "package_dir", "results_path", "timestamp", "run_config", "error",
]


def latest_package(prefix: str) -> Path | None:
    """Return the most recently created package dir matching '<prefix>-<stamp>'."""
    candidates = [
        p for p in PACKAGES_DIR.glob(f"{prefix}-*")
        if p.is_dir() and re.fullmatch(rf"{re.escape(prefix)}-\d{{8}}-\d{{4}}", p.name)
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]


def resolve_dataset(rel: str) -> Path | None:
    path = DATASETS_DIR / rel
    if path.exists():
        return path
    # fall back: exact filename search anywhere under datasets/
    matches = list(DATASETS_DIR.rglob(Path(rel).name))
    if matches:
        return matches[0]
    # fall back: case-insensitive filename search (handles upper/lower-case differences)
    target = Path(rel).name.lower()
    for p in DATASETS_DIR.rglob("*.json"):
        if p.name.lower() == target:
            return p
    return None


def newest_results_json(package_dir: Path, since: float) -> Path | None:
    """Find the results.json written by the run that just finished."""
    candidates = [
        p for p in package_dir.rglob("results.json") if p.stat().st_mtime >= since - 5
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_report(results_path: Path) -> dict:
    """Pull metrics from report.txt beside results.json, falling back to JSON."""
    out: dict = {}
    report = results_path.parent / "report.txt"
    if report.exists():
        text = report.read_text(encoding="utf-8", errors="replace")

        def grab(pattern, cast=float):
            m = re.search(pattern, text)
            return cast(m.group(1)) if m else None

        out["questions"] = grab(r"Questions run:\s+(\d+)", int)
        out["scored"] = grab(r"Scored questions:\s+(\d+)", int)
        out["accuracy"] = grab(r"Execution Accuracy:\s+([\d.]+)%")
        out["macro_f1"] = grab(r"Macro F1:\s+([\d.]+)")
        out["macro_precision"] = grab(r"Macro Precision:\s+([\d.]+)")
        out["macro_recall"] = grab(r"Macro Recall:\s+([\d.]+)")
        out["svr_initial"] = grab(r"SVR initial:\s+([\d.]+)%")
        out["svr_final"] = grab(r"SVR final:\s+([\d.]+)%")
        out["execution_success"] = grab(r"Execution Success:\s+([\d.]+)%")
        out["correction_rate"] = grab(r"Correction Rate:\s+([\d.]+)%")
        out["avg_iterations"] = grab(r"Avg Iterations:\s+([\d.]+)")
        out["avg_latency_ms"] = grab(r"Avg Latency:\s+([\d.]+) ms")

    if out.get("accuracy") is None:
        # last resort: compute rough EA from results.json
        try:
            data = json.loads(results_path.read_text(encoding="utf-8"))
            rows = data if isinstance(data, list) else data.get("results", [])
            scored = [r for r in rows if r.get("status") == "scored"]
            exact = sum(1 for r in scored if r.get("exact_match"))
            if scored:
                out["questions"] = len(rows)
                out["scored"] = len(scored)
                out["accuracy"] = round(exact / len(scored) * 100, 1)
        except Exception:
            pass
    return out


def append_row(row: dict) -> None:
    RUNS_CSV.parent.mkdir(parents=True, exist_ok=True)
    new_file = not RUNS_CSV.exists()
    with RUNS_CSV.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


# Standard run configuration (the paper's reference configuration):
#   --chunking class_based --k 5 --abox-rag --abox-k 5
CHUNKING_STRATEGY = "class_based"
RETRIEVAL_K = "5"
ABOX_K = "5"
USE_ABOX_RAG = True


def run_one(ontology: str, package_dir: Path, dataset: Path, model: str, dry: bool) -> dict:
    cmd = [
        sys.executable, "evaluate.py",
        "--dataset", str(dataset),
        "--package", str(package_dir),
        "--model", model,
        "--chunking", CHUNKING_STRATEGY,
        "--k", RETRIEVAL_K,
    ]
    if USE_ABOX_RAG:
        cmd += ["--abox-rag", "--abox-k", ABOX_K]
    print(f"  $ {' '.join(cmd)}")
    if dry:
        return {"status": "dry-run"}

    started = time.time()
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    except Exception as exc:
        return {"status": "launch_failed", "error": str(exc),
                "wall_seconds": round(time.time() - started, 1)}

    wall = round(time.time() - started, 1)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return {"status": "failed", "wall_seconds": wall,
                "error": tail[-1][:300] if tail else f"exit {proc.returncode}"}

    results_path = newest_results_json(package_dir, started)
    if results_path is None:
        return {"status": "no_results", "wall_seconds": wall,
                "error": "run finished but no results.json found"}

    row = {"status": "ok", "wall_seconds": wall, "results_path": str(results_path)}
    row.update(parse_report(results_path))
    row["run_config"] = (
        f"chunking={CHUNKING_STRATEGY} k={RETRIEVAL_K} "
        f"abox_rag={USE_ABOX_RAG} abox_k={ABOX_K}"
    )
    return row


def write_matrix(models: list[str]) -> None:
    if not RUNS_CSV.exists():
        print("No runs recorded yet.")
        return
    with RUNS_CSV.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("status") == "ok"]
    if not rows:
        print("No successful runs to summarise.")
        return

    by_key = {(r["ontology"], r["model"]): r for r in rows}
    ontologies = sorted({r["ontology"] for r in rows})

    def question_count(ont: str) -> str:
        """Scored-question count for this ontology (same dataset across models,
        so take it from whichever run recorded it first)."""
        for m in models:
            r = by_key.get((ont, m))
            if r and r.get("scored"):
                return r["scored"]
        for m in models:
            r = by_key.get((ont, m))
            if r and r.get("questions"):
                return r["questions"]
        return ""

    with MATRIX_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        header = ["ontology", "questions"]
        for m in models:
            header += [f"{m} EA%", f"{m} F1", f"{m} latency_ms"]
        writer.writerow(header)
        for ont in ontologies:
            line = [ont, question_count(ont)]
            for m in models:
                r = by_key.get((ont, m))
                line += ([r.get("accuracy", ""), r.get("macro_f1", ""),
                          r.get("avg_latency_ms", "")] if r else ["", "", ""])
            writer.writerow(line)

    ont_width = max(len(o) for o in ontologies) + 2
    q_width = 6
    width = ont_width + q_width
    print("\n" + "=" * (width + 22 * len(models)))
    print("MATRIX — Execution Accuracy % (Macro F1)")
    print("=" * (width + 22 * len(models)))
    print("ontology".ljust(ont_width) + "n".ljust(q_width)
          + "".join(m.ljust(22) for m in models))
    print("-" * (width + 22 * len(models)))
    for ont in ontologies:
        line = ont.ljust(ont_width) + str(question_count(ont)).ljust(q_width)
        for m in models:
            r = by_key.get((ont, m))
            cell = f"{r['accuracy']}% ({r['macro_f1']})" if r else "-"
            line += cell.ljust(22)
        print(line)
    print(f"\nPer-run detail: {RUNS_CSV.name}\nMatrix:         {MATRIX_CSV.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="subset of ontology keys")
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--matrix-only", action="store_true",
                    help="rebuild the matrix from the existing runs CSV")
    args = ap.parse_args()

    if args.matrix_only:
        write_matrix(args.models)
        return

    keys = args.only or list(ONTOLOGIES)
    unknown = [k for k in keys if k not in ONTOLOGIES]
    if unknown:
        sys.exit(f"Unknown ontology key(s): {', '.join(unknown)}")

    # Resolve everything up front so missing files surface before any long run.
    plan, problems = [], []
    for key in keys:
        prefix, ds_rel = ONTOLOGIES[key]
        pkg = latest_package(prefix)
        ds = resolve_dataset(ds_rel)
        if pkg is None:
            problems.append(f"{key}: no package matching '{prefix}-*'")
        elif ds is None:
            problems.append(f"{key}: dataset not found ({ds_rel})")
        else:
            plan.append((key, pkg, ds))

    if problems:
        print("Skipping (unresolved):")
        for p in problems:
            print(f"  ! {p}")
        print()
    if not plan:
        sys.exit("Nothing to run.")

    total = len(plan) * len(args.models)
    print(f"Ontologies: {len(plan)}  Models: {len(args.models)}  Runs: {total}")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}\n")

    done = 0
    for key, pkg, ds in plan:
        print(f"### {key}  [{pkg.name}]")
        if not args.dry_run:
            act = subprocess.run(
                [sys.executable, "activate.py", "--package", str(pkg)],
                cwd=ROOT, capture_output=True, text=True,
            )
            if act.returncode != 0:
                print(f"  ACTIVATION FAILED — skipping ontology")
                for model in args.models:
                    done += 1
                    append_row({"ontology": key, "model": model, "status": "activation_failed",
                                "package_dir": str(pkg),
                                "timestamp": datetime.now().isoformat(timespec="seconds"),
                                "error": (act.stderr or "").strip()[-300:]})
                continue
            print("  activated")

        for model in args.models:
            done += 1
            print(f"[{done}/{total}] {key} | {model}")
            row = run_one(key, pkg, ds, model, args.dry_run)
            row.update({
                "ontology": key, "model": model, "package_dir": str(pkg),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            })
            if not args.dry_run:
                append_row(row)
                if row["status"] == "ok":
                    print(f"      EA={row.get('accuracy')}%  F1={row.get('macro_f1')}  "
                          f"lat={row.get('avg_latency_ms')}ms  ({row['wall_seconds']}s)")
                else:
                    print(f"      {row['status'].upper()}: {row.get('error', '')}")

    if not args.dry_run:
        write_matrix(args.models)


if __name__ == "__main__":
    main()