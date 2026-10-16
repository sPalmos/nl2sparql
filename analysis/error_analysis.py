"""Error analysis for the paper.

Two stages.

STAGE 1 (automatic, all models)
    python analysis/error_analysis.py

    Uses exactly the runs that fill the main table (same selection rule as
    build_headline_tables.py: newest reference-configuration run per
    ontology x model, empty-gold questions excluded, ontologies with fewer
    than 5 questions excluded). For every failed question it assigns

      outcome category (one per failure, checked in this order)
        E1 invalid_query     no query passed validation / no query produced
        E2 execution_error   query valid but the endpoint rejected it or timed out
        E3 empty_result      query ran, returned nothing, gold is non-empty
        E4 wrong_answer      query returned rows, none of them correct
        E5 too_few_rows      every returned row is correct, some gold rows missing
        E6 too_many_rows     every gold row returned, plus extra rows
        E7 partial_overlap   some correct rows, some missing, some extra

      structural flags (several per failure; gold query vs generated query)
        aggregate_diff, group_by_diff, limit_diff, duplicate_rows,
        negation_diff, missing_gold_term (a property or instance URI used by
        the gold query that the generated query never mentions),
        missing_gold_class,
        missing_gold_literal, column_count_diff

    Outputs (output/tables/)
      error_cases_all.csv              every failure, all models, with both queries
      table_error_categories.csv       outcome categories per model (counts and %)
      table_error_flags.csv            structural flags per model (% of failures)
      error_sample_for_annotation.csv  stratified sample of qwen2.5-coder:32b
                                       failures with empty columns for manual labels

STAGE 2 (after manual annotation)
    python analysis/error_analysis.py --summarize output/tables/error_sample_for_annotation.csv

    Reads the annotated sample and writes output/tables/table_error_taxonomy.csv
    (root-cause counts, %, and 95% Wilson intervals).

The manual root-cause labels are defined in ERROR_ANNOTATION_GUIDE.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_headline_tables import MODEL_ORDER, matches_reference, ontology_key  # noqa: E402

ANNOTATED_MODEL = "qwen2.5-coder:32b"
MIN_QUESTIONS = 5
OUTCOMES = [
    ("E1", "invalid_query"), ("E2", "execution_error"), ("E3", "empty_result"),
    ("E4", "wrong_answer"), ("E5", "too_few_rows"), ("E6", "too_many_rows"),
    ("E7", "partial_overlap"),
]
FLAGS = [
    "aggregate_diff", "group_by_diff", "limit_diff", "duplicate_rows",
    "negation_diff", "missing_gold_term", "missing_gold_class",
    "missing_gold_literal", "column_count_diff",
]
ROOT_CAUSES = {
    "A": "schema linking",
    "B": "entity linking",
    "C": "graph pattern / join path",
    "D": "filtering and negation",
    "E": "aggregation, grouping, ordering",
    "F": "projection / answer form",
    "G": "gold or benchmark issue",
    "H": "validity / execution",
    "I": "other",
}


# --------------------------------------------------------------------------
# Run selection (mirrors build_headline_tables.py)
# --------------------------------------------------------------------------

def select_runs(root: Path) -> dict[tuple[str, str], Path]:
    chosen: dict[tuple[str, str], tuple[str, Path]] = {}
    for path in sorted(root.rglob("results.json")):
        try:
            results = json.loads(path.read_text(encoding="utf-8"))
        except Exception:                                          # noqa: BLE001
            continue
        if not results.get("results"):
            continue
        if not matches_reference(results.get("pipeline_config") or {}):
            continue
        key = (ontology_key(results.get("package_dir", "")), results.get("model_name", ""))
        stamp = results.get("timestamp", "")
        if key not in chosen or stamp > chosen[key][0]:
            chosen[key] = (stamp, path)
    return {key: path for key, (_, path) in chosen.items()}


# --------------------------------------------------------------------------
# Query feature extraction
# --------------------------------------------------------------------------

_AGG = re.compile(r"\b(COUNT|SUM|AVG|MIN|MAX|GROUP_CONCAT|SAMPLE)\s*\(", re.I)
_PREFIXED = re.compile(r"(?<![\w?$<])([A-Za-z][\w-]*)?:([A-Za-z_][\w.-]*)")
_IRI = re.compile(r"<([^>\s]+)>")
_LITERAL = re.compile(r'"([^"]{2,})"')


def _body(query: str) -> str:
    """Query text without PREFIX lines and comments."""
    lines = [l for l in (query or "").splitlines()
             if not l.strip().upper().startswith(("PREFIX", "BASE", "#"))]
    return "\n".join(lines)


def _local(term: str) -> str:
    return re.split(r"[#/:]", term.strip("<>"))[-1].lower()


def _terms(query: str) -> tuple[set[str], set[str]]:
    """(predicate-or-class local names, class local names) mentioned in the query."""
    body = _body(query)
    names = {_local(m.group(2)) for m in _PREFIXED.finditer(body)}
    names |= {_local(m.group(1)) for m in _IRI.finditer(body)}
    classes = set()
    for match in re.finditer(r"(?:\ba\b|rdf:type|<http://www\.w3\.org/1999/02/22-rdf-syntax-ns#type>)\s+(\S+)", body):
        classes.add(_local(match.group(1).strip("<>;.,)}")))
    names -= {"type", "label"}
    return names, classes


def _features(query: str) -> dict:
    body = _body(query)
    upper = body.upper()
    return {
        "aggregates": sorted({m.group(1).upper() for m in _AGG.finditer(body)}),
        "group_by": "GROUP BY" in upper,
        "limit": bool(re.search(r"\bLIMIT\b", upper)),
        "negation": bool(re.search(r"NOT\s+EXISTS|\bMINUS\b|!=", upper)),
        "literals": {s.lower() for s in _LITERAL.findall(body)},
    }


def structural_flags(gold: str, generated: str, gold_rows: list, gen_rows) -> list[str]:
    flags = []
    g, p = _features(gold), _features(generated or "")
    if g["aggregates"] != p["aggregates"]:
        flags.append("aggregate_diff")
    if g["group_by"] != p["group_by"]:
        flags.append("group_by_diff")
    if g["limit"] != p["limit"]:
        flags.append("limit_diff")
    if g["negation"] != p["negation"]:
        flags.append("negation_diff")
    gold_names, gold_classes = _terms(gold)
    gen_names, gen_classes = _terms(generated or "")
    gen_text = (generated or "").lower()
    if any(name not in gen_names and name not in gen_text for name in gold_names - gold_classes):
        flags.append("missing_gold_term")
    if any(c not in gen_names and c not in gen_text for c in gold_classes):
        flags.append("missing_gold_class")
    if any(lit not in gen_text for lit in g["literals"]):
        flags.append("missing_gold_literal")
    if gold_rows and gen_rows:
        if len(gold_rows[0]) != len(gen_rows[0]):
            flags.append("column_count_diff")
        keys = [json.dumps(row, sort_keys=True) for row in gen_rows]
        if len(set(keys)) < len(keys):
            flags.append("duplicate_rows")
    return flags


# --------------------------------------------------------------------------
# Outcome category
# --------------------------------------------------------------------------

def outcome(question: dict) -> str:
    comparison = question.get("comparison") or {}
    iterations = question.get("iterations") or []
    last = iterations[-1] if iterations else {}
    stages = last.get("validation_stages") or {}
    if not (question.get("final_sparql") or "").strip() or (stages and not all(stages.values())):
        return "E1"
    if (last.get("execution_status") or "EXECUTION_OK") != "EXECUTION_OK" \
            or question.get("final_answers") is None:
        return "E2"
    if comparison.get("generated_is_empty"):
        return "E3"
    tp = comparison.get("true_positives", 0) or 0
    precision = comparison.get("precision", 0.0) or 0.0
    recall = comparison.get("recall", 0.0) or 0.0
    if tp == 0:
        return "E4"
    if precision >= 0.999 and recall < 0.999:
        return "E5"
    if recall >= 0.999 and precision < 0.999:
        return "E6"
    return "E7"


# --------------------------------------------------------------------------
# Stage 1
# --------------------------------------------------------------------------

def stage_one(root: Path, out_dir: Path, sample_size: int, seed: int) -> None:
    runs = select_runs(root)
    print(f"  {len(runs)} reference cells selected")
    question_counts: dict[str, int] = {}
    cases = []
    totals: Counter = Counter()

    for (ontology, model), path in sorted(runs.items()):
        results = json.loads(path.read_text(encoding="utf-8"))
        scored = [q for q in results.get("results", [])
                  if q.get("is_scored", True) and (q.get("gold_answers") or [])]
        if len(scored) < MIN_QUESTIONS:
            continue
        question_counts[ontology] = len(scored)
        totals[model] += len(scored)
        for q in scored:
            if (q.get("comparison") or {}).get("exact_match"):
                continue
            comparison = q.get("comparison") or {}
            iterations = q.get("iterations") or []
            last_error = ""
            if q.get("errors"):
                last_error = str(q["errors"][-1])[:300]
            elif iterations and iterations[-1].get("validation_errors"):
                last_error = str(iterations[-1]["validation_errors"][-1])[:300]
            cases.append({
                "model": model, "ontology": ontology, "question_id": q.get("question_id"),
                "question": q.get("nl_question", ""),
                "outcome": outcome(q),
                "flags": " ".join(structural_flags(q.get("gold_sparql", ""), q.get("final_sparql", ""),
                                                   q.get("gold_answers") or [], q.get("final_answers"))),
                "attempts": q.get("total_iterations", len(iterations)),
                "gold_rows": comparison.get("gold_size", ""),
                "generated_rows": comparison.get("generated_size", ""),
                "precision": round(comparison.get("precision", 0.0) or 0.0, 3),
                "recall": round(comparison.get("recall", 0.0) or 0.0, 3),
                "last_error": last_error,
                "gold_sparql": _body(q.get("gold_sparql", "")).strip(),
                "generated_sparql": _body(q.get("final_sparql", "")).strip(),
                "gold_answers_head": json.dumps((q.get("gold_answers") or [])[:5], ensure_ascii=False),
                "generated_answers_head": json.dumps((q.get("final_answers") or [])[:5], ensure_ascii=False),
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(cases[0].keys()) if cases else []
    with (out_dir / "error_cases_all.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(cases)

    models = [m for m in MODEL_ORDER if totals[m]] + sorted(m for m in totals if m not in MODEL_ORDER)
    by_model = defaultdict(list)
    for case in cases:
        by_model[case["model"]].append(case)

    # Outcome categories per model
    with (out_dir / "table_error_categories.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "questions", "failures"]
                        + [f"{code} {name} n" for code, name in OUTCOMES]
                        + [f"{code} {name} %" for code, name in OUTCOMES])
        for model in models:
            failures = by_model[model]
            counts = Counter(c["outcome"] for c in failures)
            n = len(failures) or 1
            writer.writerow([model, totals[model], len(failures)]
                            + [counts[code] for code, _ in OUTCOMES]
                            + [round(100 * counts[code] / n, 1) for code, _ in OUTCOMES])

    # Structural flags per model
    with (out_dir / "table_error_flags.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "failures"] + [f"{flag} %" for flag in FLAGS])
        for model in models:
            failures = by_model[model]
            n = len(failures) or 1
            writer.writerow([model, len(failures)]
                            + [round(100 * sum(flag in c["flags"].split() for c in failures) / n, 1)
                               for flag in FLAGS])

    # Stratified annotation sample for the annotated model
    target = by_model.get(ANNOTATED_MODEL, [])
    rng = random.Random(seed)
    per_ontology = defaultdict(list)
    for case in target:
        per_ontology[case["ontology"]].append(case)
    sample = []
    if target:
        for ontology, items in sorted(per_ontology.items()):
            quota = max(1, round(sample_size * len(items) / len(target)))
            sample.extend(rng.sample(items, min(quota, len(items))))
        rng.shuffle(sample)
    annotation_fields = (["case_id", "ontology", "question_id", "question", "outcome", "flags",
                          "manual_root_cause", "gold_correct", "note"]
                         + ["attempts", "gold_rows", "generated_rows", "last_error",
                            "gold_sparql", "generated_sparql",
                            "gold_answers_head", "generated_answers_head"])
    with (out_dir / "error_sample_for_annotation.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=annotation_fields, extrasaction="ignore")
        writer.writeheader()
        for index, case in enumerate(sample, start=1):
            writer.writerow({**case, "case_id": index, "manual_root_cause": "",
                             "gold_correct": "", "note": ""})

    # Console summary
    print(f"\n{'model':22s}{'questions':>10s}{'failures':>10s}  "
          + "".join(f"{code:>6s}" for code, _ in OUTCOMES))
    for model in models:
        counts = Counter(c["outcome"] for c in by_model[model])
        n = len(by_model[model]) or 1
        print(f"{model:22s}{totals[model]:>10d}{len(by_model[model]):>10d}  "
              + "".join(f"{100 * counts[code] / n:>5.0f}%" for code, _ in OUTCOMES))
    print("\n  " + "  ".join(f"{code}={name}" for code, name in OUTCOMES))
    print(f"\n  {len(target)} {ANNOTATED_MODEL} failures; {len(sample)} sampled for annotation")
    print(f"  wrote {out_dir / 'error_cases_all.csv'}")
    print(f"        {out_dir / 'table_error_categories.csv'}")
    print(f"        {out_dir / 'table_error_flags.csv'}")
    print(f"        {out_dir / 'error_sample_for_annotation.csv'}")


# --------------------------------------------------------------------------
# Stage 2
# --------------------------------------------------------------------------

def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z = 1.96
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * max(0.0, centre - half), 100 * min(1.0, centre + half)


def stage_two(annotated: Path, out_dir: Path) -> None:
    with annotated.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    labelled = [r for r in rows if r.get("manual_root_cause", "").strip()]
    missing = len(rows) - len(labelled)
    bad = [r["case_id"] for r in labelled
           if r["manual_root_cause"].strip().upper()[:1] not in ROOT_CAUSES]
    if bad:
        sys.exit(f"Unknown root-cause code in case(s) {', '.join(bad)}; use one of {', '.join(ROOT_CAUSES)}")
    counts = Counter(r["manual_root_cause"].strip().upper()[:1] for r in labelled)
    by_outcome = defaultdict(Counter)
    for r in labelled:
        by_outcome[r["outcome"]][r["manual_root_cause"].strip().upper()[:1]] += 1
    gold_wrong = sum(r.get("gold_correct", "").strip().lower() in ("n", "no", "0", "false")
                     for r in labelled)

    n = len(labelled)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "table_error_taxonomy.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["code", "root_cause", "n", "percent", "ci95_low", "ci95_high"]
                        + [f"within {code}" for code, _ in OUTCOMES])
        for code, name in ROOT_CAUSES.items():
            low, high = wilson(counts[code], n)
            writer.writerow([code, name, counts[code], round(100 * counts[code] / n, 1) if n else "",
                             round(low, 1), round(high, 1)]
                            + [by_outcome[o][code] for o, _ in OUTCOMES])

    print(f"\n  annotated {n} cases ({missing} left blank and ignored)")
    for code, name in ROOT_CAUSES.items():
        low, high = wilson(counts[code], n)
        print(f"  {code} {name:34s}{counts[code]:>4d}  {100 * counts[code] / max(n, 1):5.1f}%  "
              f"[{low:.1f}, {high:.1f}]")
    print(f"\n  gold judged incorrect: {gold_wrong} of {n}")
    print(f"  wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--out-dir", default=str(ROOT / "output" / "tables"))
    parser.add_argument("--sample", type=int, default=80, help="annotation sample size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--summarize", help="annotated sample CSV (stage 2)")
    args = parser.parse_args()
    if args.summarize:
        stage_two(Path(args.summarize), Path(args.out_dir))
    else:
        stage_one(Path(args.root).resolve(), Path(args.out_dir), args.sample, args.seed)


if __name__ == "__main__":
    main()