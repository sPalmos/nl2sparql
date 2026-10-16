"""Automatic difficulty level per question, computed from its gold SPARQL.

  P  number of triple patterns in the WHERE clause (including those inside
     OPTIONAL, UNION, MINUS, NOT EXISTS and subqueries), rdf:type / `a`
     patterns excluded, every pattern after `;` or `,` counted separately.
     Counted on rdflib's parse tree of the query, not on the text.
  O  number of DISTINCT operations used, out of:
     aggregation (COUNT/SUM/AVG/MIN/MAX), GROUP BY, ORDER BY, LIMIT/OFFSET,
     FILTER, negation (NOT EXISTS / MINUS), UNION, OPTIONAL, subquery, HAVING.
     Detected from keywords after removing comments, string literals and IRIs.
     FILTER NOT EXISTS counts as negation only; any other FILTER counts as FILTER.

  simple   P <= SIMPLE_P and O <= SIMPLE_O
  complex  P >= COMPLEX_P or O >= COMPLEX_O
  medium   everything else

Questions: the scored, non-empty-gold questions of the main-table runs
(20 ontologies; 1,042 questions), same selection as build_headline_tables.py.

Two modes, run in this order:
    python analysis/difficulty_levels.py                  # counts only, no accuracy
    python analysis/difficulty_levels.py --ea             # after the thresholds are fixed
Thresholds:  --simple-p 2 --simple-o 1 --complex-p 5 --complex-o 3  (defaults, fixed 2026-10-08)

Outputs (output/tables/)
    difficulty_profile.csv        every question: P, O, operations, level, old tier
    table_difficulty_levels.csv   (--ea only) pooled lenient EA per level, 6 models
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rdflib.namespace import RDF
from rdflib.plugins.sparql.parser import parseQuery
from rdflib.plugins.sparql.parserutils import CompValue

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_headline_tables import (EXCLUDED_ONTOLOGIES, MODEL_ORDER,  # noqa: E402
                                   matches_reference, ontology_key)

LEVELS = ("simple", "medium", "complex")
KW = r"(?<![?$\w:])"           # not part of a variable or prefixed name
OPERATIONS = {
    "aggregation": KW + r"(COUNT|SUM|AVG|MIN|MAX)\s*\(",
    "group_by":    KW + r"GROUP\s+BY\b",
    "order_by":    KW + r"ORDER\s+BY\b",
    "limit_offset": KW + r"(LIMIT|OFFSET)\b",
    "filter":      KW + r"FILTER\b(?!\s*NOT\s+EXISTS)",
    "negation":    KW + r"(NOT\s+EXISTS|MINUS)\b",
    "union":       KW + r"UNION\b",
    "optional":    KW + r"OPTIONAL\b",
    "having":      KW + r"HAVING\b",
}


# --------------------------------------------------------------------------
# Query analysis
# --------------------------------------------------------------------------

def strip_query(query: str) -> str:
    """Remove comments, string literals and IRIs so keywords inside them are ignored."""
    text = re.sub(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', '""', query)
    text = re.sub(r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'', '""', text)
    text = re.sub(r"<[^<>\s]*>", "<>", text)
    text = re.sub(r"#[^\n]*", "", text)
    return text


def operations(query: str) -> list[str]:
    text = strip_query(query)
    found = [name for name, pattern in OPERATIONS.items() if re.search(pattern, text, re.I)]
    if len(re.findall(KW + r"SELECT\b", text, re.I)) > 1:
        found.append("subquery")
    return found


def declare_missing_prefixes(query: str) -> str:
    text = strip_query(query)
    declared = {m.lower() for m in re.findall(r"PREFIX\s+([\w-]*):", text, re.I)}
    used = set(re.findall(r"(?<![\w?$<])([A-Za-z][\w-]*|):(?=[\w-]|\s)", text))
    missing = [p for p in used if p.lower() not in declared and p.lower() not in ("http", "https", "urn", "mailto")]
    header = "".join(f"PREFIX {p}: <http://undeclared.example/{p or 'base'}#>\n" for p in sorted(missing))
    return header + query


def _is_rdf_type(predicate) -> bool:
    """True for `a`, rdf:type or <...rdf-syntax-ns#type>, unwrapping rdflib's path wrappers."""
    node = predicate
    for wrapper in ("PathAlternative", "PathSequence", "PathElt"):
        if isinstance(node, CompValue) and node.name == wrapper:
            part = dict.get(node, "part")
            if isinstance(part, list):
                if len(part) != 1:
                    return False
                part = part[0]
            node = part
    if node == RDF.type:
        return True
    return (isinstance(node, CompValue) and node.name == "pname"
            and dict.get(node, "prefix") == "rdf" and dict.get(node, "localname") == "type")


def triple_patterns(query: str) -> int:
    """Triple patterns anywhere in the parsed query (nested groups, OPTIONAL, UNION,
    MINUS, EXISTS / NOT EXISTS, subqueries), rdf:type excluded.

    Counted on rdflib's parse tree, where every `;` / `,` is already expanded: each
    TriplesBlock entry is a flat list s, p, o, s, p, o, ...
    """
    tree = parseQuery(declare_missing_prefixes(query))
    count = 0

    def walk(node) -> None:
        nonlocal count
        if isinstance(node, CompValue):
            if node.name == "TriplesBlock":
                for flat in dict.get(node, "triples", []):
                    flat = list(flat)
                    count += sum(1 for i in range(1, len(flat), 3) if not _is_rdf_type(flat[i]))
                return
            for key in dict.keys(node):
                walk(dict.__getitem__(node, key))
        elif isinstance(node, (list, tuple)) or type(node).__name__ == "ParseResults":
            for value in node:
                walk(value)

    walk(list(tree))
    return count


def level(p: int, o: int, args) -> str:
    if p >= args.complex_p or o >= args.complex_o:
        return "complex"
    if p <= args.simple_p and o <= args.simple_o:
        return "simple"
    return "medium"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def main_table_runs() -> dict[tuple[str, str], Path]:
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
    return {k: p for k, (_, p) in chosen.items()}


def dataset_questions() -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for path in (ROOT / "evaluation" / "datasets").rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:                                          # noqa: BLE001
            continue
        if isinstance(data, dict) and "questions" in data:
            out[data.get("dataset_name", path.stem)] = {
                str(q.get("id", q.get("question_id"))): q for q in data["questions"]}
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simple-p", type=int, default=2)
    parser.add_argument("--simple-o", type=int, default=1)
    parser.add_argument("--complex-p", type=int, default=5)
    parser.add_argument("--complex-o", type=int, default=3)
    parser.add_argument("--ea", action="store_true", help="also compute accuracy (only after thresholds are fixed)")
    args = parser.parse_args()

    runs = main_table_runs()
    questions_by_dataset = dataset_questions()
    models = [m for m in MODEL_ORDER if any(m == mm for _, mm in runs)]

    # Profile every scored, non-empty-gold question once (from the first model's run).
    profile: dict[tuple[str, str], dict] = {}
    results_cache: dict[tuple[str, str], dict] = {}
    failures = []
    for (ontology, model), path in sorted(runs.items()):
        results = json.loads(path.read_text(encoding="utf-8"))
        results_cache[(ontology, model)] = results
        dataset = questions_by_dataset.get(results.get("dataset_name", ""), {})
        for q in results["results"]:
            if not q.get("is_scored", True) or not (q.get("gold_answers") or []):
                continue
            key = (ontology, str(q.get("question_id")))
            if key in profile:
                continue
            source = dataset.get(key[1], {})
            gold = source.get("gold_sparql") or q.get("gold_sparql") or ""
            try:
                p = triple_patterns(gold)
                status = "ok"
            except Exception as exc:                               # noqa: BLE001
                p, status = -1, f"parse error: {str(exc)[:80]}"
                failures.append((ontology, key[1], status))
            ops = operations(gold)
            profile[key] = {"ontology": ontology, "question_id": key[1], "P": p, "O": len(ops),
                            "operations": " ".join(ops), "old_tier": source.get("complexity_tier", ""),
                            "parse_status": status}

    for row in profile.values():
        row["level"] = level(row["P"], row["O"], args) if row["P"] >= 0 else "unparsed"

    out = ROOT / "output" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    rows = sorted(profile.values(), key=lambda r: (r["ontology"], r["question_id"]))
    with (out / "difficulty_profile.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # ---- counts (no accuracy) -------------------------------------------------
    n = len(rows)
    print(f"Thresholds: simple P<={args.simple_p} and O<={args.simple_o} | complex P>={args.complex_p} or O>={args.complex_o}")
    print(f"Scored questions: {n}   (parse failures: {len(failures)})\n")
    levels = Counter(r["level"] for r in rows)
    for name in LEVELS + ("unparsed",):
        if levels[name]:
            print(f"  {name:9s} {levels[name]:5d}  ({100 * levels[name] / n:.1f}%)")
    print("\nP distribution:  " + "  ".join(f"{k}:{v}" for k, v in sorted(Counter(r['P'] for r in rows).items())))
    print("O distribution:  " + "  ".join(f"{k}:{v}" for k, v in sorted(Counter(r['O'] for r in rows).items())))
    print("Joint P x O (rows P, columns O):")
    os_ = sorted({r["O"] for r in rows})
    joint = Counter((r["P"], r["O"]) for r in rows)
    print("      " + "".join(f"{o:>6d}" for o in os_))
    for p in sorted({r["P"] for r in rows}):
        print(f"  {p:>3d} " + "".join(f"{joint[(p, o)]:>6d}" for o in os_))
    print("\nOperations (questions using each):")
    for name, count in Counter(op for r in rows for op in r["operations"].split()).most_common():
        print(f"  {name:13s} {count}")
    print("\nNew level x old complexity_tier (sanity check, not for the paper):")
    tiers = ["simple", "medium", "complex"]
    cross = Counter((r["level"], r["old_tier"]) for r in rows)
    print(f"  {'':10s}" + "".join(f"{t:>9s}" for t in tiers))
    for lv in LEVELS:
        print(f"  {lv:10s}" + "".join(f"{cross[(lv, t)]:>9d}" for t in tiers))
    for o, q, s in failures[:20]:
        print(f"  ! {o} {q}: {s}")
    print("\nwrote output/tables/difficulty_profile.csv")

    if not args.ea:
        print("\n(accuracy not computed; rerun with --ea once the thresholds are fixed)")
        return

    # ---- accuracy (only with --ea) ---------------------------------------------
    hits, counts = defaultdict(Counter), defaultdict(Counter)
    for (ontology, model), results in results_cache.items():
        for q in results["results"]:
            key = (ontology, str(q.get("question_id")))
            if key not in profile:
                continue
            correct = bool((q.get("comparison") or {}).get("exact_match"))
            for lv in (profile[key]["level"], "all"):
                counts[lv][model] += 1
                hits[lv][model] += correct
    table = []
    for lv in ("all",) + LEVELS:
        row = {"label": "all" if lv == "all" else "difficulty_level", "value": lv,
               "questions_scored": counts[lv][models[0]], "questions_total": counts[lv][models[0]]}
        for m in models:
            row[f"{m} EA"] = round(100.0 * hits[lv][m] / counts[lv][m], 1) if counts[lv][m] else ""
        row["note"] = (f"simple: P<={args.simple_p} and O<={args.simple_o}; complex: P>={args.complex_p} or O>={args.complex_o}"
                       if lv == "all" else "")
        table.append(row)
    with (out / "table_difficulty_levels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print("\nPooled lenient EA per level:")
    for r in table:
        print(f"  {r['value']:8s} {r['questions_scored']:5d}  " + "  ".join(f"{r[f'{m} EA']!s:>5s}" for m in models))
    print("wrote output/tables/table_difficulty_levels.csv")


if __name__ == "__main__":
    main()