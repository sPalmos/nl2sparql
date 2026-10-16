"""Run the ablation study from a named experiment registry.

Every configuration is declared once, given a name, and executed through
`evaluate.py` with explicit flags. Nothing is hand-edited between runs, and each
run's flags are recorded in its own `run_config.json`, so any row in the results
can be traced back to the exact system that produced it.

Two families of configuration are declared:

  ladder      S0..S4, built up by architectural dependency. Answers "how much
              does each component add as the system is assembled?"
  loo         leave-one-out from the full system. Answers "what is each
              component worth inside the finished system?"

Results are appended to results_ablation.csv after every run, so an interrupted
sweep loses at most the run in flight.

Usage
    python run_ablations.py --list
    python run_ablations.py --dry-run
    python run_ablations.py --family ladder
    python run_ablations.py --configs S0 S4 loo_no_abox
    python run_ablations.py --summary            # rebuild tables from the CSV
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

ROOT = Path(__file__).resolve().parent
PACKAGES_DIR = ROOT / "ontology_packages"
DATASETS_DIR = ROOT / "evaluation" / "datasets"
RUNS_CSV = ROOT / "results_ablation.csv"
SUMMARY_CSV = ROOT / "results_ablation_summary.csv"

MODEL = "qwen2.5-coder:32b"

# Ablation subset: spans difficulty, size, naming style and namespace structure.
# (The paper's subset also includes eNOVATION, which is not public.)
SUBSET = {
    "bestiary":                            ("bestiary",                            "bestiary_sandro_eval_dataset.json"),
    "ck25":                                ("ck25",                                "ck25_text2sparql_combined_eval_dataset.json"),
    "spider-dog-kennels":                  ("spider-dog-kennels",                  "spider4sparql_sandro/spider_dog_kennels_combined_eval_dataset.json"),
    "spider-student-transcripts-tracking": ("spider-student-transcripts-tracking", "spider4sparql_sandro/spider_student_transcripts_tracking_combined_eval_dataset.json"),
    "spider-world-1":                      ("spider-world-1",                      "spider4sparql_sandro/spider_world_1_combined_eval_dataset.json"),
}

# Reference configuration = the full system, matching the main matrix.
REFERENCE = {
    "chunking": "class_based",
    "k": "5",
    "abox_rag": True,
    "abox_k": "5",
    "max_attempts": "3",
}


def cfg(description: str, family: str, **overrides) -> dict:
    return {"description": description, "family": family, "flags": overrides}


# --------------------------------------------------------------------------- #
# Experiment registry
# --------------------------------------------------------------------------- #
REGISTRY: dict[str, dict] = {
    # ---- cumulative ladder, ordered by architectural dependency ----
    "S0": cfg(
        "Question and prefixes only: no schema retrieval, no rule blocks, "
        "no validation loop, no instance grounding",
        "ladder",
        no_tbox_retrieval=True, abox_rag=False, minimal_prompt_rules=True,
        no_vocab_validation=True, max_attempts="1",
    ),
    "S1": cfg(
        "+ TBox retrieval (schema chunks reach the prompt)",
        "ladder",
        abox_rag=False, minimal_prompt_rules=True,
        no_vocab_validation=True, max_attempts="1",
    ),
    "S2": cfg(
        "+ prompt rule blocks and adaptive naming strategy",
        "ladder",
        abox_rag=False, no_vocab_validation=True, max_attempts="1",
    ),
    "S3": cfg(
        "+ formal validation and execution-guided correction",
        "ladder",
        abox_rag=False,
    ),
    "S4": cfg(
        "+ ABox retrieval with reranking and gating = full system",
        "ladder",
    ),

    # ---- leave-one-out from the full system ----
    "loo_no_abox": cfg(
        "Full system without ABox retrieval",
        "loo",
        abox_rag=False,
    ),
    "loo_no_abox_rerank": cfg(
        "Full system, instance candidates in raw FAISS order",
        "loo",
        no_abox_rerank=True,
    ),
    "loo_no_abox_gating": cfg(
        "Full system, weak instance candidates no longer suppressed",
        "loo",
        no_abox_gating=True,
    ),
    "loo_no_correction": cfg(
        "Full system, single generation attempt",
        "loo",
        max_attempts="1",
    ),
    "loo_no_adaptive_naming": cfg(
        "Full system assuming rdfs:label everywhere",
        "loo",
        no_adaptive_naming=True,
    ),
    "loo_no_vocab_validation": cfg(
        "Full system without the vocabulary-membership stage",
        "loo",
        no_vocab_validation=True,
    ),
    "loo_minimal_rules": cfg(
        "Full system with prefix and constraint rules only",
        "loo",
        minimal_prompt_rules=True,
    ),

    # ---- retrieval configuration sweeps ----
    # The framework offers three schema chunking schemes and tunable retrieval
    # depth; the paper describes them, so they are measured rather than assumed.
    "chunk_property_based": cfg(
        "Full system with property-based schema chunks",
        "retrieval",
        chunking="property_based",
    ),
    "chunk_composite": cfg(
        "Full system with composite schema chunks",
        "retrieval",
        chunking="composite",
    ),
    "tbox_k3": cfg(
        "Full system retrieving 3 schema chunks instead of 5",
        "retrieval",
        k="3",
    ),
    "tbox_k10": cfg(
        "Full system retrieving 10 schema chunks instead of 5",
        "retrieval",
        k="10",
    ),
    "abox_k3": cfg(
        "Full system retrieving 3 instance candidates instead of 5",
        "retrieval",
        abox_k="3",
    ),
    "abox_k10": cfg(
        "Full system retrieving 10 instance candidates instead of 5",
        "retrieval",
        abox_k="10",
    ),
    "reactive_abox": cfg(
        "Full system plus the legacy reactive ABox discovery on empty results",
        "retrieval",
        reactive_abox_discovery=True,
    ),

    # ---- determinism check (same config, repeated) ----
    # ---- naive baseline: whole schema in the prompt, no TBox selection ----
    # retrieve_context clamps k to the index size, so k=1000 returns every
    # class chunk. Prompts beyond num_ctx are truncated by Ollama; that is part
    # of what this baseline measures.
    "full_schema": cfg(
        "Full system with every schema chunk in the prompt (no schema selection)",
        "baseline",
        k="1000",
    ),

    "det_full_a": cfg("Determinism check, repetition A", "determinism"),
    "det_full_b": cfg("Determinism check, repetition B", "determinism"),
    "det_full_c": cfg("Determinism check, repetition C", "determinism"),
}

CSV_FIELDS = [
    "config", "family", "ontology", "model", "status",
    "questions", "scored", "accuracy", "macro_f1",
    "svr_initial", "svr_final", "execution_success",
    "correction_rate", "avg_iterations", "avg_latency_ms",
    "wall_seconds", "flags", "package_dir", "results_path", "timestamp", "error",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def latest_package(prefix: str) -> Path | None:
    candidates = [
        p for p in PACKAGES_DIR.glob(f"{prefix}-*")
        if p.is_dir() and re.fullmatch(rf"{re.escape(prefix)}-\d{{8}}-\d{{4}}", p.name)
    ]
    return sorted(candidates, key=lambda p: p.name)[-1] if candidates else None


def resolve_dataset(rel: str) -> Path | None:
    path = DATASETS_DIR / rel
    if path.exists():
        return path
    matches = list(DATASETS_DIR.rglob(Path(rel).name))
    if matches:
        return matches[0]
    target = Path(rel).name.lower()
    for candidate in DATASETS_DIR.rglob("*.json"):
        if candidate.name.lower() == target:
            return candidate
    return None


def build_command(dataset: Path, package: Path, flags: dict) -> list[str]:
    """Turn a registry entry into a concrete evaluate.py command."""
    merged = dict(REFERENCE)
    merged.update(flags)

    cmd = [
        sys.executable, "evaluate.py",
        "--dataset", str(dataset),
        "--package", str(package),
        "--model", MODEL,
        "--chunking", merged["chunking"],
        "--k", str(merged["k"]),
        "--max-attempts", str(merged["max_attempts"]),
    ]
    if merged.get("abox_rag", True):
        cmd += ["--abox-rag", "--abox-k", str(merged["abox_k"])]
    for flag, argument in (
        ("reactive_abox_discovery", "--reactive-abox-discovery"),
        ("no_tbox_retrieval", "--no-tbox-retrieval"),
        ("no_abox_rerank", "--no-abox-rerank"),
        ("no_abox_gating", "--no-abox-gating"),
        ("no_adaptive_naming", "--no-adaptive-naming"),
        ("minimal_prompt_rules", "--minimal-prompt-rules"),
        ("no_vocab_validation", "--no-vocab-validation"),
    ):
        if merged.get(flag):
            cmd.append(argument)
    return cmd


def newest_results_json(package_dir: Path, since: float) -> Path | None:
    candidates = [
        p for p in package_dir.rglob("results.json")
        if p.stat().st_mtime >= since - 5
    ]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def parse_report(results_path: Path) -> dict:
    out: dict = {}
    report = results_path.parent / "report.txt"
    if not report.exists():
        return out
    text = report.read_text(encoding="utf-8", errors="replace")

    def grab(pattern, cast=float):
        match = re.search(pattern, text)
        return cast(match.group(1)) if match else None

    out["questions"] = grab(r"Questions run:\s+(\d+)", int)
    out["scored"] = grab(r"Scored questions:\s+(\d+)", int)
    out["accuracy"] = grab(r"Execution Accuracy:\s+([\d.]+)%")
    out["macro_f1"] = grab(r"Macro F1:\s+([\d.]+)")
    out["svr_initial"] = grab(r"SVR initial:\s+([\d.]+)%")
    out["svr_final"] = grab(r"SVR final:\s+([\d.]+)%")
    out["execution_success"] = grab(r"Execution Success:\s+([\d.]+)%")
    out["correction_rate"] = grab(r"Correction Rate:\s+([\d.]+)%")
    out["avg_iterations"] = grab(r"Avg Iterations:\s+([\d.]+)")
    out["avg_latency_ms"] = grab(r"Avg Latency:\s+([\d.]+) ms")
    return out


def append_row(row: dict) -> None:
    new_file = not RUNS_CSV.exists()
    with RUNS_CSV.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in CSV_FIELDS})


def already_done(config: str, ontology: str) -> bool:
    if not RUNS_CSV.exists():
        return False
    with RUNS_CSV.open(encoding="utf-8") as handle:
        return any(
            r["config"] == config and r["ontology"] == ontology and r["status"] == "ok"
            for r in csv.DictReader(handle)
        )


def summarise() -> None:
    if not RUNS_CSV.exists():
        print("No ablation runs recorded yet.")
        return
    with RUNS_CSV.open(encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["status"] == "ok"]
    if not rows:
        print("No successful ablation runs.")
        return

    ontologies = sorted({r["ontology"] for r in rows})
    configs = [name for name in REGISTRY if any(r["config"] == name for r in rows)]
    cells = {(r["config"], r["ontology"]): r for r in rows}

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["config", "family", "description", "mean_EA"] + ontologies)
        for name in configs:
            values = [
                float(cells[(name, o)]["accuracy"])
                for o in ontologies
                if (name, o) in cells and cells[(name, o)]["accuracy"]
            ]
            mean = round(sum(values) / len(values), 1) if values else ""
            writer.writerow(
                [name, REGISTRY[name]["family"], REGISTRY[name]["description"], mean]
                + [cells[(name, o)]["accuracy"] if (name, o) in cells else ""
                   for o in ontologies]
            )
    print(f"  wrote {SUMMARY_CSV.name}")

    width = max(len(c) for c in configs) + 2
    print("\n" + "=" * (width + 10 + 14 * len(ontologies)))
    print("ABLATION SUMMARY — Execution Accuracy %")
    print("=" * (width + 10 + 14 * len(ontologies)))
    print("config".ljust(width) + "mean".ljust(10)
          + "".join(o[:12].ljust(14) for o in ontologies))
    print("-" * (width + 10 + 14 * len(ontologies)))
    for name in configs:
        values = [
            float(cells[(name, o)]["accuracy"])
            for o in ontologies
            if (name, o) in cells and cells[(name, o)]["accuracy"]
        ]
        mean = f"{sum(values) / len(values):.1f}%" if values else "-"
        line = name.ljust(width) + mean.ljust(10)
        for ontology in ontologies:
            cell = cells.get((name, ontology))
            line += ((cell["accuracy"] + "%") if cell else "-").ljust(14)
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", help="named configurations to run")
    parser.add_argument("--family", nargs="*",
                        choices=["ladder", "loo", "retrieval", "determinism"])
    parser.add_argument("--only", nargs="*", help="restrict to these ontologies")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--redo", action="store_true", help="re-run configurations already recorded")
    args = parser.parse_args()

    if args.list:
        for name, entry in REGISTRY.items():
            print(f"{name:26s} [{entry['family']:11s}] {entry['description']}")
        return
    if args.summary:
        summarise()
        return

    names = args.configs or [
        n for n, e in REGISTRY.items()
        if not args.family or e["family"] in args.family
    ]
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        sys.exit(f"Unknown configuration(s): {', '.join(unknown)}")

    ontologies = args.only or list(SUBSET)
    plan: list[tuple[str, str, Path, Path]] = []
    problems: list[str] = []
    for ontology in ontologies:
        if ontology not in SUBSET:
            problems.append(f"{ontology}: not in the ablation subset")
            continue
        prefix, dataset_rel = SUBSET[ontology]
        package = latest_package(prefix)
        dataset = resolve_dataset(dataset_rel)
        if package is None:
            problems.append(f"{ontology}: no package matching '{prefix}-*'")
        elif dataset is None:
            problems.append(f"{ontology}: dataset not found ({dataset_rel})")
        else:
            for name in names:
                plan.append((name, ontology, package, dataset))

    if problems:
        print("Skipping (unresolved):")
        for problem in problems:
            print(f"  ! {problem}")
        print()
    if not plan:
        sys.exit("Nothing to run.")

    print(f"Configurations: {len(names)}  Ontologies: {len(ontologies)}  "
          f"Runs: {len(plan)}")
    print(f"Model: {MODEL}")
    print(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}\n")

    done = 0
    current_package: Path | None = None
    for name, ontology, package, dataset in plan:
        done += 1
        if not args.redo and already_done(name, ontology):
            print(f"[{done}/{len(plan)}] {name} | {ontology} — already recorded, skipping")
            continue

        entry = REGISTRY[name]
        command = build_command(dataset, package, entry["flags"])
        print(f"[{done}/{len(plan)}] {name} | {ontology}")
        print(f"    {entry['description']}")
        if args.dry_run:
            print(f"    $ {' '.join(command)}")
            continue

        if package != current_package:
            activation = subprocess.run(
                [sys.executable, "activate.py", "--package", str(package)],
                cwd=ROOT, capture_output=True, text=True,
            )
            if activation.returncode != 0:
                print("    ACTIVATION FAILED — skipping")
                append_row({
                    "config": name, "family": entry["family"], "ontology": ontology,
                    "model": MODEL, "status": "activation_failed",
                    "flags": json.dumps(entry["flags"]), "package_dir": str(package),
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "error": (activation.stderr or "").strip()[-300:],
                })
                continue
            current_package = package

        started = time.time()
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        wall = round(time.time() - started, 1)

        row = {
            "config": name, "family": entry["family"], "ontology": ontology,
            "model": MODEL, "wall_seconds": wall,
            "flags": json.dumps(entry["flags"]), "package_dir": str(package),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        if process.returncode != 0:
            tail = (process.stderr or process.stdout or "").strip().splitlines()
            row.update({"status": "failed",
                        "error": tail[-1][:300] if tail else f"exit {process.returncode}"})
            print(f"    FAILED: {row['error']}")
        else:
            results_path = newest_results_json(package, started)
            if results_path is None:
                row.update({"status": "no_results",
                            "error": "run finished but no results.json found"})
                print("    NO RESULTS")
            else:
                row.update({"status": "ok", "results_path": str(results_path)})
                row.update(parse_report(results_path))
                print(f"    EA={row.get('accuracy')}%  F1={row.get('macro_f1')}  ({wall}s)")
        append_row(row)

    if not args.dry_run:
        summarise()


if __name__ == "__main__":
    main()