"""Build the publication tables from completed runs.

Scores every run under two rules, both using the evaluator's own normalisation
so the analysis cannot drift from the pipeline:

  lenient  the rule the evaluator applies: surface normalisation, URI <-> label
           alias resolution from the package's source graph, and acceptance of a
           generated row that carries every gold value plus extra columns
  strict   identical, except a generated row must match a gold row exactly;
           extra columns are no longer accepted

Alias resolution is kept in BOTH rules. Resolving an answer URI to the entity's
label is not leniency: a query answering with `:TC1_UCLouvain_CTMA` has named the
right training centre. The two rules therefore isolate exactly one question —
whether returning the entity URI alongside its label counts as correct.

Empty-gold questions are excluded throughout, and only runs matching the
reference configuration enter the main matrix.

Outputs
  table_main_matrix.csv    ontologies x models, primary rule
  table_sensitivity.csv    per cell: both rules and the gap between them
  table_model_summary.csv  per model: accuracy, correction contribution, latency
  table_excluded_runs.csv  runs outside the reference configuration

Usage
    python build_headline_tables.py --root . --out-dir analysis
    python build_headline_tables.py --primary strict
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation.answer_comparison import normalize_result_set
from evaluation.experiment_runner import answer_aliases_from_package, prefix_map_from_package

REFERENCE_CONFIG = {
    "chunking_strategy": "class_based",
    "retrieval_top_k": 5,
    "use_abox_rag": True,
    "abox_retrieval_top_k": 5,
    "use_reactive_abox_discovery": False,
    "correction_max_iterations": 3,
}

# Component switches added for the ablation study. Runs recorded before the
# switches existed do not carry them, and those runs had every component on,
# so a missing key counts as True. Without this check, ablation runs that share
# the retrieval settings above (for example the run without gating) would be
# taken as the newest reference run for qwen2.5-coder:32b.
REFERENCE_FLAGS = (
    "use_tbox_retrieval", "use_abox_rerank", "use_abox_gating",
    "use_adaptive_naming", "use_full_prompt_rules", "use_vocabulary_validation",
)

MODEL_ORDER = [
    "qwen2.5-coder:7b", "qwen2.5-coder:32b", "qwen3:8b",
    "qwen3-coder:30b", "llama3.1:8b", "deepseek-r1:32b",
]

EXCLUDED_ONTOLOGIES = {"spider-real-estate-properties": "only 2 evaluation questions"}

_alias_cache: dict[str, dict] = {}
_prefix_cache: dict[str, dict] = {}


def package_context(package_dir: str):
    """Alias map and prefix map for a package.

    Building the alias map parses the package's source ontology, which is slow
    for large graphs, so the result is cached in the package as
    `answer_aliases.cache.json` and reused on later runs. Delete that file to
    force a rebuild.
    """
    if package_dir in _alias_cache:
        return _alias_cache[package_dir], _prefix_cache[package_dir]

    cache_file = Path(package_dir) / "answer_aliases.cache.json"
    aliases = None
    if cache_file.exists():
        try:
            aliases = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:                                          # noqa: BLE001
            aliases = None

    if aliases is None:
        print(f"    building alias map for {Path(package_dir).name} ...",
              end="", flush=True)
        try:
            aliases = answer_aliases_from_package(package_dir)
        except Exception as exc:                                   # noqa: BLE001
            print(f" failed ({exc})")
            aliases = {}
        else:
            print(f" {len(aliases)} entities")
            try:
                cache_file.write_text(json.dumps(aliases), encoding="utf-8")
            except Exception:                                      # noqa: BLE001
                pass

    try:
        prefixes = prefix_map_from_package(package_dir)
    except Exception:                                              # noqa: BLE001
        prefixes = {}

    _alias_cache[package_dir] = aliases
    _prefix_cache[package_dir] = prefixes
    return aliases, prefixes


def ontology_key(package_dir: str) -> str:
    """Package folder name without its timestamp: spider-car-1-20261010-2127(-2) -> spider-car-1.

    Onboarding appends -YYYYMMDD-HHMM, plus -N when that name already exists."""
    name = Path(str(package_dir).replace("\\", "/")).name
    match = re.match(r"^(.*?)-\d{8}-\d{4}(?:-\d+)?$", name)
    return match.group(1) if match else name


def matches_reference(pipeline: dict) -> bool:
    return (all(pipeline.get(key) == value for key, value in REFERENCE_CONFIG.items())
            and all(pipeline.get(flag, True) is True for flag in REFERENCE_FLAGS))


def score_run(results: dict, package_dir: str) -> dict:
    """Score one run under both rules.

    The lenient figures are taken from the comparison the evaluator already
    stored, so they are exactly the pipeline's own numbers. The strict figure is
    recomputed only for questions the evaluator marked correct, because strict
    correctness implies lenient correctness: a question the lenient rule rejects
    cannot pass the stricter one. Those questions always have small result sets,
    so the recomputation never touches the pathological cases (an unconstrained
    query can return tens of thousands of rows, and comparing them row by row is
    quadratic).
    """
    scored = empty_gold = 0
    lenient = strict = 0
    lenient_f1 = 0.0
    lenient_first_attempt = strict_first_attempt = 0
    latencies: list[float] = []
    aliases = prefixes = None

    for question in results.get("results", []):
        if not question.get("is_scored", True):
            continue
        gold = question.get("gold_answers") or []
        latencies.append(float(question.get("total_latency_ms", 0.0) or 0.0))
        if not gold:
            empty_gold += 1
            continue

        scored += 1
        comparison = question.get("comparison") or {}
        first_attempt = question.get("total_iterations", 0) <= 1
        lenient_f1 += float(comparison.get("f1", 0.0) or 0.0)

        if not comparison.get("exact_match"):
            continue
        lenient += 1
        lenient_first_attempt += first_attempt

        generated = question.get("final_answers")
        if generated is None:
            continue
        if aliases is None:
            aliases, prefixes = package_context(package_dir)
        if normalize_result_set(gold, prefixes, aliases) == normalize_result_set(
                generated, prefixes, aliases):
            strict += 1
            strict_first_attempt += first_attempt

    def pct(count: int):
        return round(100.0 * count / scored, 1) if scored else ""

    return {
        "n": scored,
        "n_empty_gold_excluded": empty_gold,
        "ea_lenient": pct(lenient),
        "f1_lenient": round(lenient_f1 / scored, 3) if scored else "",
        "ea_strict": pct(strict),
        "f1_strict": "",
        "ea_lenient_first_attempt": pct(lenient_first_attempt),
        "ea_strict_first_attempt": pct(strict_first_attempt),
        "correction_gain_lenient": (
            round(100.0 * (lenient - lenient_first_attempt) / scored, 1) if scored else ""),
        "correction_gain_strict": (
            round(100.0 * (strict - strict_first_attempt) / scored, 1) if scored else ""),
        "avg_latency_ms": round(statistics.mean(latencies)) if latencies else "",
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        print(f"  (nothing to write for {path.name})")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path.name}  ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--out-dir", default="analysis")
    parser.add_argument("--primary", choices=["lenient", "strict"], default="lenient",
                        help="which rule fills the main matrix (default: lenient)")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    primary, secondary = args.primary, "strict" if args.primary == "lenient" else "lenient"

    # Pass 1: pick the newest run per (ontology, model). Scoring is deferred so
    # that alias maps are only ever built for packages that actually win a cell.
    chosen: dict[tuple[str, str], tuple[str, Path, str]] = {}
    excluded: list[dict] = []
    scanned = 0

    for results_path in sorted(root.rglob("results.json")):
        try:
            results = json.loads(results_path.read_text(encoding="utf-8"))
        except Exception:                                          # noqa: BLE001
            continue
        if not results.get("results"):
            continue
        scanned += 1
        pipeline = results.get("pipeline_config") or {}
        package_dir = results.get("package_dir", "")
        ontology, model = ontology_key(package_dir), results.get("model_name", "")
        timestamp = results.get("timestamp", "")

        # API models are run twice on 19 ontologies and reported separately
        # by api_comparison.py, so they stay out of the six-model tables.
        if model.startswith("api:"):
            continue

        if not matches_reference(pipeline):
            excluded.append({
                "ontology": ontology, "model": model, "timestamp": timestamp,
                "chunking": pipeline.get("chunking_strategy", ""),
                "tbox_k": pipeline.get("retrieval_top_k", ""),
                "abox_rag": pipeline.get("use_abox_rag", ""),
                "max_iterations": pipeline.get("correction_max_iterations", ""),
                "run_dir": str(results_path.parent),
            })
            continue

        key = (ontology, model)
        if key not in chosen or timestamp > chosen[key][0]:
            chosen[key] = (timestamp, results_path, package_dir)

    print(f"  scanned {scanned} runs; {len(chosen)} cells to score, "
          f"{len(excluded)} outside the reference configuration\n")

    # Pass 2: score the winners.
    cells: dict[tuple[str, str], dict] = {}
    for index, (key, (timestamp, results_path, package_dir)) in enumerate(
            sorted(chosen.items()), start=1):
        ontology, model = key
        print(f"  [{index}/{len(chosen)}] {ontology} x {model}")
        try:
            results = json.loads(results_path.read_text(encoding="utf-8"))
        except Exception:                                          # noqa: BLE001
            continue
        row = score_run(results, package_dir)
        row.update({"ontology": ontology, "model": model, "timestamp": timestamp})
        cells[key] = row
    print()

    if not cells:
        print("No runs matched the reference configuration.")
        return

    ontologies = sorted({o for o, _ in cells})
    reported = [o for o in ontologies if o not in EXCLUDED_ONTOLOGIES]
    models = [m for m in MODEL_ORDER if any(m == mm for _, mm in cells)]
    models += sorted({mm for _, mm in cells if mm not in MODEL_ORDER})

    matrix_rows = []
    for ontology in reported:
        any_cell = next((cells[(ontology, m)] for m in models if (ontology, m) in cells), None)
        row = {"ontology": ontology,
               "n": any_cell["n"] if any_cell else "",
               "empty_gold_excluded": any_cell["n_empty_gold_excluded"] if any_cell else ""}
        for model in models:
            cell = cells.get((ontology, model))
            row[f"{model} EA"] = cell[f"ea_{primary}"] if cell else ""
            row[f"{model} F1"] = cell[f"f1_{primary}"] if cell else ""
        matrix_rows.append(row)
    write_csv(out_dir / "table_main_matrix.csv", matrix_rows)

    sensitivity = []
    for ontology in ontologies:
        for model in models:
            cell = cells.get((ontology, model))
            if not cell:
                continue
            gap = ""
            if cell["ea_lenient"] != "" and cell["ea_strict"] != "":
                gap = round(cell["ea_lenient"] - cell["ea_strict"], 1)
            sensitivity.append({
                "ontology": ontology, "model": model, "n": cell["n"],
                "ea_lenient": cell["ea_lenient"], "ea_strict": cell["ea_strict"],
                "extra_column_effect_pts": gap,
                "f1_lenient": cell["f1_lenient"],
                "in_headline_table": ontology not in EXCLUDED_ONTOLOGIES,
            })
    write_csv(out_dir / "table_sensitivity.csv", sensitivity)

    def pooled(values, field):
        total = sum(v["n"] for v in values if isinstance(v[field], (int, float)))
        hits = sum(round(v[field] * v["n"] / 100) for v in values
                   if isinstance(v[field], (int, float)))
        return round(100 * hits / total, 1) if total else ""

    summary = []
    for model in models:
        values = [cells[(o, model)] for o in reported if (o, model) in cells]
        if not values:
            continue

        def mean_of(field: str, digits: int = 1):
            numbers = [v[field] for v in values if isinstance(v[field], (int, float))]
            return round(statistics.mean(numbers), digits) if numbers else ""

        summary.append({
            "model": model, "ontologies": len(values),
            f"mean_ea_{primary}": mean_of(f"ea_{primary}"),
            f"mean_ea_{secondary}": mean_of(f"ea_{secondary}"),
            "extra_column_effect_pts": round(mean_of("ea_lenient") - mean_of("ea_strict"), 1),
            "mean_f1_lenient": mean_of("f1_lenient", 3),
            "mean_ea_first_attempt": mean_of(f"ea_{primary}_first_attempt"),
            "correction_gain_pts": mean_of(f"correction_gain_{primary}"),
            "mean_latency_ms": mean_of("avg_latency_ms"),
            # Pooled (micro) EA: all scored questions together, so large
            # ontologies weigh more. The macro mean above is the headline.
            "questions_pooled": sum(v["n"] for v in values),
            f"pooled_ea_{primary}": pooled(values, f"ea_{primary}"),
            f"pooled_ea_{secondary}": pooled(values, f"ea_{secondary}"),
        })
    write_csv(out_dir / "table_model_summary.csv", summary)
    write_csv(out_dir / "table_excluded_runs.csv", excluded)

    width = max(len(o) for o in reported) + 2
    print("\n" + "=" * (width + 8 + 20 * len(models)))
    print(f"MAIN MATRIX — Execution Accuracy % (Macro F1), {primary} rule, empty-gold excluded")
    print("=" * (width + 8 + 20 * len(models)))
    print("ontology".ljust(width) + "n".ljust(8) + "".join(m.ljust(20) for m in models))
    print("-" * (width + 8 + 20 * len(models)))
    for row in matrix_rows:
        line = row["ontology"].ljust(width) + str(row["n"]).ljust(8)
        for model in models:
            ea, f1 = row.get(f"{model} EA", ""), row.get(f"{model} F1", "")
            cell_text = f"{ea}% ({f1})" if f1 not in ("", None) else (f"{ea}%" if ea != "" else "-")
            line += cell_text.ljust(20)
        print(line)

    print("\n" + "=" * 96)
    print("MODEL SUMMARY (mean across reported ontologies)")
    print("=" * 96)
    print(f"{'model':22s}{'EA ' + primary:>12s}{'EA ' + secondary:>12s}"
          f"{'extra-col':>11s}{'F1':>8s}{'1st att':>10s}{'correction':>12s}{'latency':>10s}")
    print("-" * 96)
    for row in summary:
        print(f"{row['model']:22s}{row['mean_ea_' + primary]:>11}%{row['mean_ea_' + secondary]:>11}%"
              f"{row['extra_column_effect_pts']:>11}{row['mean_f1_lenient']:>8}"
              f"{row['mean_ea_first_attempt']:>9}%{row['correction_gain_pts']:>12}"
              f"{row['mean_latency_ms']:>10}")
    for name, reason in EXCLUDED_ONTOLOGIES.items():
        print(f"\nExcluded from the headline table:\n  {name}: {reason}")
    if excluded:
        print(f"\n{len(excluded)} run(s) outside the reference configuration -> table_excluded_runs.csv")


if __name__ == "__main__":
    main()