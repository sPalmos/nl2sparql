"""API model vs the best local model, on the ontologies both were run on.

The API model was run twice with the reference configuration (it is not
deterministic: the provider samples at its default temperature). The local
reference model is deterministic and enters with its main-table run.

Run selection (same rules as build_headline_tables.py)
  - only runs matching the reference configuration
  - API model:   the two newest runs per ontology (older smoke tests ignored)
  - local model: the newest run per ontology (the main-table cell)
  - empty-gold questions excluded; ontologies with < 5 scored questions excluded

Outputs (output/tables/)
  table_api_per_ontology.csv   EA per ontology: run 1, run 2, mean, local model
  table_api_summary.csv        macro / pooled EA, F1, first attempt, correction, latency
  table_api_significance.csv   paired tests: local vs each API run, run 1 vs run 2

Statistics
  McNemar (exact) per API run against the local model, Holm-adjusted over the
  two runs; stratified paired bootstrap CI on the macro delta of the run mean
  (a question solved in one of the two runs counts 0.5); run-to-run stability
  as the number of questions whose correctness flips between the two runs.

Usage (repository root)
    python analysis/api_comparison.py
    python analysis/api_comparison.py --api-model api:gemini-3.1-flash-lite --local-model qwen2.5-coder:32b
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_headline_tables import EXCLUDED_ONTOLOGIES, matches_reference, ontology_key  # noqa: E402
from significance_tests import StrictScorer, bootstrap_macro_ci, holm, mcnemar_exact  # noqa: E402

MIN_QUESTIONS = 5


def collect_runs(root: Path, models: set[str]) -> dict[tuple[str, str], list[tuple[str, Path]]]:
    runs: dict[tuple[str, str], list[tuple[str, Path]]] = {}
    for path in sorted(root.rglob("results.json")):
        try:
            results = json.loads(path.read_text(encoding="utf-8"))
        except Exception:                                          # noqa: BLE001
            continue
        if not results.get("results") or results.get("model_name") not in models:
            continue
        if not matches_reference(results.get("pipeline_config") or {}):
            continue
        key = (ontology_key(results.get("package_dir", "")), results["model_name"])
        runs.setdefault(key, []).append((results.get("timestamp", ""), path))
    return {k: sorted(v) for k, v in runs.items()}


def load(path: Path, strict: StrictScorer) -> dict:
    """Per-question outcomes and run-level figures for one results.json."""
    results = json.loads(path.read_text(encoding="utf-8"))
    package_dir = results.get("package_dir", "")
    questions, latencies = {}, []
    for q in results.get("results", []):
        if not q.get("is_scored", True):
            continue
        latencies.append(float(q.get("total_latency_ms", 0.0) or 0.0))
        if not (q.get("gold_answers") or []):
            continue
        comparison = q.get("comparison") or {}
        lenient = bool(comparison.get("exact_match"))
        questions[q["question_id"]] = {
            "lenient": lenient,
            "strict": lenient and strict.correct(q, package_dir),
            "f1": float(comparison.get("f1", 0.0) or 0.0),
            "first": q.get("total_iterations", 0) <= 1,
        }
    return {"q": questions, "latency": statistics.mean(latencies) if latencies else 0.0,
            "timestamp": results.get("timestamp", ""), "path": str(path)}


def ea(run: dict, rule: str, ids: list[str], first_only: bool = False) -> float:
    hits = sum(1 for i in ids if run["q"][i][rule] and (run["q"][i]["first"] or not first_only))
    return 100.0 * hits / len(ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--out-dir", default=str(ROOT / "output" / "tables"))
    parser.add_argument("--api-model", default="api:gemini-3.1-flash-lite")
    parser.add_argument("--local-model", default="qwen2.5-coder:32b")
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root, out_dir = Path(args.root).resolve(), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = collect_runs(root, {args.api_model, args.local_model})
    strict = StrictScorer()

    data: dict[str, dict] = {}
    for (ontology, model), found in sorted(runs.items()):
        if model != args.api_model or ontology in EXCLUDED_ONTOLOGIES:
            continue
        if len(found) < 2:
            print(f"  ! {ontology}: only {len(found)} API run(s), skipped")
            continue
        if (ontology, args.local_model) not in runs:
            print(f"  ! {ontology}: no {args.local_model} run, skipped")
            continue
        api1, api2 = (load(p, strict) for _, p in found[-2:])
        local = load(runs[(ontology, args.local_model)][-1][1], strict)
        ids = sorted(set(api1["q"]) & set(api2["q"]) & set(local["q"]))
        if len(ids) < MIN_QUESTIONS:
            continue
        if not (set(api1["q"]) == set(api2["q"]) == set(local["q"])):
            print(f"  ! {ontology}: question sets differ; using {len(ids)} shared")
        data[ontology] = {"api1": api1, "api2": api2, "local": local, "ids": ids}
        print(f"  {ontology:38s} n={len(ids):>4}  API runs {api1['timestamp'][:16]} / "
              f"{api2['timestamp'][:16]}  local {local['timestamp'][:16]}")

    if not data:
        sys.exit("No ontology has two API runs and a local run.")
    ontologies = sorted(data)

    # ---- per ontology --------------------------------------------------------
    per_rows = []
    for o in ontologies:
        d, ids = data[o], data[o]["ids"]
        flips = sum(1 for i in ids if d["api1"]["q"][i]["lenient"] != d["api2"]["q"][i]["lenient"])
        row = {"ontology": o, "n": len(ids)}
        for rule in ("lenient", "strict"):
            r1, r2 = ea(d["api1"], rule, ids), ea(d["api2"], rule, ids)
            row[f"api_run1_ea_{rule}"] = round(r1, 1)
            row[f"api_run2_ea_{rule}"] = round(r2, 1)
            row[f"api_mean_ea_{rule}"] = round((r1 + r2) / 2, 1)
            row[f"local_ea_{rule}"] = round(ea(d["local"], rule, ids), 1)
        row["delta_lenient_pp"] = round(row["api_mean_ea_lenient"] - row["local_ea_lenient"], 1)
        row["api_flips_between_runs"] = flips
        per_rows.append(row)

    # ---- summary -------------------------------------------------------------
    def summarise(label: str, runs_of: list[str]) -> dict:
        def macro(fn) -> float:
            return statistics.mean(statistics.mean(fn(data[o][r], data[o]["ids"]) for r in runs_of)
                                   for o in ontologies)
        pooled_n = sum(len(data[o]["ids"]) for o in ontologies)

        def pooled(rule: str) -> float:
            hits = sum(statistics.mean(sum(data[o][r]["q"][i][rule] for i in data[o]["ids"])
                                       for r in runs_of) for o in ontologies)
            return round(100.0 * hits / pooled_n, 1)

        lenient = macro(lambda run, ids: ea(run, "lenient", ids))
        first = macro(lambda run, ids: ea(run, "lenient", ids, first_only=True))
        return {
            "system": label, "ontologies": len(ontologies), "questions_pooled": pooled_n,
            "mean_ea_lenient": round(lenient, 1),
            "mean_ea_strict": round(macro(lambda run, ids: ea(run, "strict", ids)), 1),
            "pooled_ea_lenient": pooled("lenient"), "pooled_ea_strict": pooled("strict"),
            "mean_f1_lenient": round(macro(lambda run, ids: statistics.mean(
                run["q"][i]["f1"] for i in ids)), 3),
            "mean_ea_first_attempt": round(first, 1),
            "correction_gain_pts": round(lenient - first, 1),
            "mean_latency_ms": round(macro(lambda run, ids: run["latency"])),
        }

    summary = [
        summarise(f"{args.api_model} (mean of 2 runs)", ["api1", "api2"]),
        summarise(f"{args.api_model} run 1", ["api1"]),
        summarise(f"{args.api_model} run 2", ["api2"]),
        summarise(args.local_model, ["local"]),
    ]

    # ---- paired tests --------------------------------------------------------
    def compare(ref: str, var: list[str], rule: str) -> dict:
        b = c = total = 0
        pairs_by_ontology, terms = {}, []
        for o in ontologies:
            d, ids = data[o], data[o]["ids"]
            pairs = [(int(d[ref]["q"][i][rule]),
                      statistics.mean(int(d[v]["q"][i][rule]) for v in var)) for i in ids]
            pairs_by_ontology[o] = pairs
            if len(var) == 1:
                b += sum(1 for r, v in pairs if r and not v)
                c += sum(1 for r, v in pairs if v and not r)
            total += len(pairs)
            terms.append(100.0 * sum(v - r for r, v in pairs) / len(pairs))
        low, high = bootstrap_macro_ci(pairs_by_ontology, args.replicates, args.seed)
        return {"reference": ref, "variant": "+".join(var), "rule": rule,
                "ontologies": len(ontologies), "questions": total,
                "macro_delta_pp": round(statistics.mean(terms), 1),
                "ref_only_b": b if len(var) == 1 else "", "variant_only_c": c if len(var) == 1 else "",
                "mcnemar_p": mcnemar_exact(b, c) if len(var) == 1 else "",
                "ci95_low_pp": round(low, 1), "ci95_high_pp": round(high, 1)}

    names = {"local": args.local_model, "api1": f"{args.api_model} run 1",
             "api2": f"{args.api_model} run 2", "api1+api2": f"{args.api_model} mean"}
    records = []
    for rule in ("lenient", "strict"):
        group = [compare("local", ["api1"], rule), compare("local", ["api2"], rule)]
        for record, adjusted in zip(group, holm([r["mcnemar_p"] for r in group])):
            record["holm_p"] = adjusted
        mean_record = compare("local", ["api1", "api2"], rule)
        mean_record["holm_p"] = ""
        stability = compare("api1", ["api2"], rule)
        stability["holm_p"] = stability["mcnemar_p"]
        records += group + [mean_record, stability]
    for r in records:
        r["reference"], r["variant"] = names[r["reference"]], names[r["variant"]]

    def write(path: Path, rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({k: (f"{v:.4g}" if isinstance(v, float) and k.endswith("_p") else v)
                                 for k, v in row.items()})

    write(out_dir / "table_api_per_ontology.csv", per_rows)
    write(out_dir / "table_api_summary.csv", summary)
    write(out_dir / "table_api_significance.csv", records)

    # ---- console -------------------------------------------------------------
    print(f"\n{'ontology':38s}{'n':>5s}{'run1':>8s}{'run2':>8s}{'mean':>8s}{'local':>8s}{'delta':>8s}{'flips':>7s}")
    for r in per_rows:
        print(f"{r['ontology']:38s}{r['n']:>5d}{r['api_run1_ea_lenient']:>8.1f}{r['api_run2_ea_lenient']:>8.1f}"
              f"{r['api_mean_ea_lenient']:>8.1f}{r['local_ea_lenient']:>8.1f}{r['delta_lenient_pp']:>+8.1f}"
              f"{r['api_flips_between_runs']:>7d}")
    print(f"\n{'system':48s}{'macro':>7s}{'strict':>8s}{'pooled':>8s}{'F1':>7s}{'1st':>7s}{'corr':>6s}{'lat ms':>8s}")
    for s in summary:
        print(f"{s['system']:48s}{s['mean_ea_lenient']:>7.1f}{s['mean_ea_strict']:>8.1f}{s['pooled_ea_lenient']:>8.1f}"
              f"{s['mean_f1_lenient']:>7.3f}{s['mean_ea_first_attempt']:>7.1f}{s['correction_gain_pts']:>6.1f}"
              f"{s['mean_latency_ms']:>8d}")
    print("\nPaired tests (delta = variant minus reference, pp)")
    for r in records:
        p = f"p={r['mcnemar_p']:.3g} Holm={r['holm_p']:.3g}" if r["mcnemar_p"] != "" else "(bootstrap only)"
        bc = f"b={r['ref_only_b']} c={r['variant_only_c']}" if r["ref_only_b"] != "" else ""
        print(f"  [{r['rule']:7s}] {r['reference']} -> {r['variant']}: {r['macro_delta_pp']:+.1f} "
              f"[{r['ci95_low_pp']:+.1f}, {r['ci95_high_pp']:+.1f}]  {bc}  {p}")
    total_flips = sum(r["api_flips_between_runs"] for r in per_rows)
    total_q = sum(r["n"] for r in per_rows)
    print(f"\nAPI run-to-run flips (lenient): {total_flips} of {total_q} questions "
          f"({100.0 * total_flips / total_q:.1f}%)")
    print(f"Wrote {out_dir / 'table_api_per_ontology.csv'}, table_api_summary.csv, table_api_significance.csv")


if __name__ == "__main__":
    main()
