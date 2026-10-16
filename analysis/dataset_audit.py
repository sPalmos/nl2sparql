"""Dataset audit for the paper (read-only apart from reloading Fuseki).

For each of the 20 reported ontologies (same run selection as the tables):

  A. Provenance of the frozen package
     - the source file recorded in the package's onboarding log
     - whether a file of the same name exists in trimmed_sources/, and whether
       the package's ontology/source.* is byte-identical to it
  B. Stratification fields in the dataset file
     - how many questions carry complexity_tier, query_shape, question_type
  C. Gold reproducibility
     - every gold_sparql is re-executed on the frozen package's Fuseki dataset
       and compared with the stored gold_answers:
         identical   same rows, same strings
         equivalent  equal after the evaluator's own normalisation
         different   differ even after normalisation
         error       the gold query failed to execute
     - empty-gold questions (stored gold_answers == []) are counted separately

Usage (repo root, nothing else running; it activates each package in turn):
    python analysis/dataset_audit.py
    python analysis/dataset_audit.py --newest-packages --only spider     # re-onboarded Spider packages
    python analysis/dataset_audit.py --skip-gold      # only A and B, no Fuseki needed

Writes output/tables/table_dataset_audit.csv and output/tables/gold_differences.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_headline_tables import matches_reference, ontology_key  # noqa: E402

FUSEKI = "http://127.0.0.1:3030"
FIELDS = ("complexity_tier", "query_shape", "question_type")
MIN_QUESTIONS = 5


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_runs() -> dict[str, dict]:
    chosen: dict[str, tuple[str, dict]] = {}
    for path in sorted(ROOT.rglob("results.json")):
        try:
            results = json.loads(path.read_text(encoding="utf-8"))
        except Exception:                                          # noqa: BLE001
            continue
        if not results.get("results") or not matches_reference(results.get("pipeline_config") or {}):
            continue
        key = ontology_key(results.get("package_dir", ""))
        stamp = results.get("timestamp", "")
        if key not in chosen or stamp > chosen[key][0]:
            chosen[key] = (stamp, results)
    return {k: v for k, (_, v) in chosen.items()}


def dataset_file(dataset_name: str) -> Path | None:
    for path in (ROOT / "evaluation" / "datasets").rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:                                          # noqa: BLE001
            continue
        if isinstance(data, dict) and data.get("dataset_name") == dataset_name:
            return path
    return None


def onboard_source(package: Path) -> str:
    for log in sorted((package / "logs").glob("*")):
        if log.is_file() and "onboard" in log.name.lower():
            text = log.read_text(encoding="utf-8", errors="replace")
            match = re.search(r"[A-Za-z]:\\[^\"'\n]+?\.(?:ttl|rdf|owl|nt)|[\w./\\-]+\.(?:ttl|rdf|owl|nt)", text)
            if match:
                return match.group(0)
    return ""


def trimmed_match(package: Path, ontology: str) -> tuple[str, str]:
    sources = [p for p in (package / "ontology").glob("source.*")]
    trimmed = [p for p in (ROOT / "trimmed_sources").glob(f"{ontology}.*")] if (ROOT / "trimmed_sources").exists() else []
    if not trimmed:
        return "no", ""
    if sources and sha256(sources[0]) == sha256(trimmed[0]):
        return "yes", "identical"
    return "yes", "differs"


def run_query(endpoint: str, query: str) -> list[dict[str, str]]:
    response = httpx.post(endpoint, data={"query": query},
                          headers={"Accept": "application/sparql-results+json"}, timeout=60)
    response.raise_for_status()
    body = response.json()
    if "boolean" in body:
        return [{"result": str(bool(body["boolean"])).lower()}]
    return [{k: v["value"] for k, v in b.items()} for b in body.get("results", {}).get("bindings", [])]


def canonical(rows) -> list:
    return sorted(json.dumps(sorted(r.items()), ensure_ascii=False) for r in rows or [])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-gold", action="store_true")
    parser.add_argument("--newest-packages", action="store_true",
                        help="audit against the newest package of each ontology instead of the one used by the runs")
    parser.add_argument("--only", nargs="*", help="restrict to ontologies whose name starts with one of these")
    args = parser.parse_args()

    if not args.skip_gold:
        from evaluation.answer_comparison import compare_results            # noqa: PLC0415
        from evaluation.experiment_runner import (                          # noqa: PLC0415
            answer_aliases_from_package, prefix_map_from_package)

    rows, differences = [], []
    for ontology, results in sorted(reference_runs().items()):
        scored = [q for q in results["results"] if q.get("is_scored", True)]
        if len([q for q in scored if q.get("gold_answers")]) < MIN_QUESTIONS:
            continue
        if args.only and not any(ontology.startswith(o) for o in args.only):
            continue
        package = Path(results["package_dir"])
        if args.newest_packages:
            package = sorted(p for p in (ROOT / "ontology_packages").glob(f"{ontology}-2026*") if p.is_dir())[-1]
        path = dataset_file(results.get("dataset_name", ""))
        questions = json.loads(path.read_text(encoding="utf-8"))["questions"] if path else []
        row = {
            "ontology": ontology, "package": package.name,
            "dataset_file": path.relative_to(ROOT).as_posix() if path else "NOT FOUND",
            "questions": len(questions),
            "onboard_source": onboard_source(package),
        }
        row["trimmed_file_exists"], row["package_source_vs_trimmed"] = trimmed_match(package, ontology)
        for field in FIELDS:
            row[f"has_{field}"] = sum(1 for q in questions if q.get(field) not in (None, ""))

        if not args.skip_gold and questions:
            subprocess.run([sys.executable, "activate.py", "--package", str(package)],
                           cwd=ROOT, capture_output=True, text=True)
            endpoint = f"{FUSEKI}/{package.name}/query"
            prefixes = prefix_map_from_package(package)
            aliases = answer_aliases_from_package(package)
            counts = {"identical": 0, "equivalent": 0, "different": 0, "error": 0, "empty_gold": 0}
            for q in questions:
                stored = q.get("gold_answers") or []
                if not stored:
                    counts["empty_gold"] += 1
                try:
                    fresh = run_query(endpoint, q["gold_sparql"])
                except Exception as exc:                                  # noqa: BLE001
                    counts["error"] += 1
                    differences.append({"ontology": ontology, "question_id": q.get("id"),
                                        "kind": "error", "detail": str(exc)[:200]})
                    continue
                if canonical(fresh) == canonical(stored):
                    counts["identical"] += 1
                elif compare_results(fresh, stored, prefix_map=prefixes, answer_aliases=aliases).exact_match \
                        and compare_results(stored, fresh, prefix_map=prefixes, answer_aliases=aliases).exact_match:
                    counts["equivalent"] += 1
                else:
                    counts["different"] += 1
                    differences.append({"ontology": ontology, "question_id": q.get("id"),
                                        "kind": "different",
                                        "detail": f"stored {len(stored)} rows, fresh {len(fresh)} rows; "
                                                  f"stored head {json.dumps(stored[:2], ensure_ascii=False)[:150]}; "
                                                  f"fresh head {json.dumps(fresh[:2], ensure_ascii=False)[:150]}"})
            row.update({f"gold_{k}": v for k, v in counts.items()})
        rows.append(row)
        print(f"  {ontology:38s} q={row['questions']:>4} "
              + " ".join(f"{f[:5]}={row[f'has_{f}']}" for f in FIELDS)
              + f"  trimmed={row['trimmed_file_exists']}/{row['package_source_vs_trimmed'] or '-'}"
              + (f"  gold: ident={row.get('gold_identical')} equiv={row.get('gold_equivalent')} "
                 f"diff={row.get('gold_different')} err={row.get('gold_error')} empty={row.get('gold_empty_gold')}"
                 if not args.skip_gold else ""))

    out = ROOT / "output" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    suffix = "_newest" if args.newest_packages else ""
    with (out / f"table_dataset_audit{suffix}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (out / f"gold_differences{suffix}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ontology", "question_id", "kind", "detail"])
        writer.writeheader()
        writer.writerows(differences)
    print(f"\nwrote output/tables/table_dataset_audit{suffix}.csv and output/tables/gold_differences{suffix}.csv "
          f"({len(differences)} differing or failing gold queries)")


if __name__ == "__main__":
    main()