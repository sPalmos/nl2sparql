"""Execution accuracy per question label, for the six local models of the main table.

Run selection is identical to build_headline_tables.py: the newest
reference-configuration run per ontology x model, API models skipped,
excluded ontologies skipped, empty-gold questions excluded. Correctness is the
evaluator's stored lenient exact_match. Accuracy is pooled over questions (not
averaged over ontologies), so each value's EA is hits / questions.

Labels (complexity_tier, query_shape, question_type) are read from the dataset
files, matched to results by question id.

Usage (repository root):  python analysis/label_breakdown.py
Writes output/tables/table_label_breakdown.csv
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_headline_tables import (EXCLUDED_ONTOLOGIES, MODEL_ORDER,  # noqa: E402
                                   matches_reference, ontology_key)

LABELS = ("complexity_tier", "query_shape", "question_type")


def label_text(value) -> str:
    if value in (None, "", []):
        return "(missing)"
    if isinstance(value, list):
        return "+".join(str(v) for v in value)
    return str(value)


def dataset_labels() -> dict[str, dict[str, dict[str, str]]]:
    """dataset_name -> question id -> {label: value}."""
    out: dict[str, dict[str, dict[str, str]]] = {}
    for path in (ROOT / "evaluation" / "datasets").rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:                                          # noqa: BLE001
            continue
        if not isinstance(data, dict) or "questions" not in data:
            continue
        out[data.get("dataset_name", path.stem)] = {
            str(q.get("id", q.get("question_id"))): {l: label_text(q.get(l)) for l in LABELS}
            for q in data["questions"]}
    return out


def main() -> None:
    chosen: dict[tuple[str, str], tuple[str, Path]] = {}
    for path in sorted(ROOT.rglob("results.json")):
        try:
            results = json.loads(path.read_text(encoding="utf-8"))
        except Exception:                                          # noqa: BLE001
            continue
        model = results.get("model_name", "")
        if not results.get("results") or model not in MODEL_ORDER:
            continue
        if not matches_reference(results.get("pipeline_config") or {}):
            continue
        key = (ontology_key(results.get("package_dir", "")), model)
        if key[0] in EXCLUDED_ONTOLOGIES:
            continue
        stamp = results.get("timestamp", "")
        if key not in chosen or stamp > chosen[key][0]:
            chosen[key] = (stamp, path)

    labels_by_dataset = dataset_labels()
    hits = defaultdict(Counter)        # (label, value) -> model -> correct
    counts = defaultdict(Counter)      # (label, value) -> model -> scored questions
    totals = Counter()                 # (label, value) -> questions incl. empty gold (one model's view)
    unmatched = Counter()
    counted_for_totals: set[str] = set()

    for (ontology, model), (_, path) in sorted(chosen.items()):
        results = json.loads(path.read_text(encoding="utf-8"))
        labels = labels_by_dataset.get(results.get("dataset_name", ""), {})
        first_view = ontology not in counted_for_totals
        counted_for_totals.add(ontology)
        for q in results["results"]:
            if not q.get("is_scored", True):
                continue
            qlabels = labels.get(str(q.get("question_id")))
            if qlabels is None:
                unmatched[ontology] += first_view
                qlabels = {l: "(unmatched)" for l in LABELS}
            if first_view:
                for l in LABELS:
                    totals[(l, qlabels[l])] += 1
            if not (q.get("gold_answers") or []):
                continue
            correct = bool((q.get("comparison") or {}).get("exact_match"))
            for l in LABELS:
                counts[(l, qlabels[l])][model] += 1
                hits[(l, qlabels[l])][model] += correct
            counts[("all", "all")][model] += 1
            hits[("all", "all")][model] += correct

    models = [m for m in MODEL_ORDER if any(m == mm for _, mm in chosen)]
    rows = []
    order = [("all", "all")] + sorted(k for k in counts if k[0] != "all")
    for key in order:
        label, value = key
        scored = {counts[key][m] for m in models}
        row = {"label": label, "value": value,
               "questions_scored": counts[key][models[0]] if models else 0,
               "questions_total": sum(v for (l, _), v in totals.items() if l == LABELS[0])
               if label == "all" else totals[key]}
        if len(scored) > 1:
            row["note"] = f"scored counts differ across models: {sorted(scored)}"
        for m in models:
            n = counts[key][m]
            row[f"{m} EA"] = round(100.0 * hits[key][m] / n, 1) if n else ""
        rows.append(row)

    out = ROOT / "output" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    fields = ["label", "value", "questions_scored", "questions_total"] + [f"{m} EA" for m in models] + ["note"]
    with (out / "table_label_breakdown.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    short = {m: m.replace("qwen2.5-coder", "q2.5c").replace("qwen3-coder", "q3c")
             .replace("deepseek-r1", "ds-r1").replace("llama3.1", "llama") for m in models}
    print(f"{'label':16s}{'value':28s}{'n':>6s}{'all':>6s}" + "".join(f"{short[m]:>11s}" for m in models))
    for r in rows:
        print(f"{r['label']:16s}{r['value'][:27]:28s}{r['questions_scored']:>6}{r['questions_total']:>6}"
              + "".join(f"{r[f'{m} EA']!s:>11s}" for m in models) + (f"  ! {r['note']}" if r.get("note") else ""))
    print(f"\ncells used: {len(chosen)} ({len({o for o, _ in chosen})} ontologies x {len(models)} models)")
    if unmatched:
        print("questions without labels (id not found in the dataset file):",
              dict(unmatched))
    print("wrote output/tables/table_label_breakdown.csv")


if __name__ == "__main__":
    main()
