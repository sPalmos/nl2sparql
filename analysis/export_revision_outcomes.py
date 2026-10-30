"""Export the per-question outcomes of the revision runs (paper-v2), eNOVATION excluded.

Run from the folder that holds the revision runs (the revision working copy):
    python analysis/export_revision_outcomes.py --out results/revision_outcomes.csv

Same columns as results/paper_outcomes.csv, plus `role` and `llm_overrides`.
Roles:
  frontier        Claude Opus 5.5, the run reported (newest per ontology)
  frontier_test   the 30-question test run made before it
  temperature     runs with NL2SPARQL_TEMPERATURE set (with top-p for qwen3:8b)
  rerun_logged    deepseek-r1:32b and qwen3:8b re-runs at the paper's settings, with logging
  neutrality      qwen2.5-coder:32b on spider-singer, before/after the paper-v1.1 patch
Rows of the eNOVATION package (including the held-out set) are never written.
Holds identifiers, the generated SPARQL and the scores only (no question text, no gold answers).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "analysis"))
import build_headline_tables as bht  # noqa: E402

FIELDS = ["run_id", "role", "ontology", "model", "timestamp", "llm_overrides", "question_id",
          "scored", "lenient", "strict", "first_attempt", "attempts", "latency_ms", "final_sparql"]
NEVER = {"enovation"}


def overrides(path: Path) -> dict:
    try:
        cfg = json.loads((path.parent / "run_config.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    o = dict(cfg.get("llm_overrides") or {})
    o.pop("NL2SPARQL_API_KEY", None)          # never present, removed to be safe
    return o


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packages", default="ontology_packages")
    ap.add_argument("--out", default="results/revision_outcomes.csv")
    args = ap.parse_args()

    runs = []
    for path in sorted(Path(args.packages).rglob("results.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if not data.get("results"):
            continue
        ont = bht.ontology_key(data.get("package_dir", ""))
        if ont in NEVER or "enovation" in (data.get("dataset_name", "") or "").lower():
            continue
        runs.append((data.get("timestamp", ""), path, data, ont))
    runs.sort(key=lambda r: r[0])

    newest_frontier = {}
    for stamp, path, data, ont in runs:
        if data.get("model_name", "").startswith("api:claude"):
            newest_frontier[ont] = stamp

    rows, roles = [], Counter()
    for stamp, path, data, ont in runs:
        model = data.get("model_name", "")
        cfg = data.get("pipeline_config") or {}
        ov = overrides(path)
        if model.startswith("api:claude"):
            role = "frontier" if stamp == newest_frontier[ont] else "frontier_test"
        elif ov.get("NL2SPARQL_TEMPERATURE"):
            role = "temperature"
        elif not bht.matches_reference(cfg):
            role = "other_config"
        elif model in ("deepseek-r1:32b", "qwen3:8b"):
            role = "rerun_logged"
        elif model == "qwen2.5-coder:32b" and ont == "spider-singer":
            role = "neutrality"
        else:
            role = "other"
        aliases = prefixes = None
        run_id = f"{Path(data.get('package_dir', '')).name}/{path.parent.name}"
        for q in data["results"]:
            gold = q.get("gold_answers") or []
            scored = int(bool(q.get("is_scored", True) and gold))
            lenient = int(bool((q.get("comparison") or {}).get("exact_match"))) if scored else 0
            strict = 0
            if lenient and q.get("final_answers") is not None:
                if aliases is None:
                    aliases, prefixes = bht.package_context(data["package_dir"])
                strict = int(bht.normalize_result_set(gold, prefixes, aliases)
                             == bht.normalize_result_set(q["final_answers"], prefixes, aliases))
            rows.append({
                "run_id": run_id, "role": role, "ontology": ont, "model": model, "timestamp": stamp,
                "llm_overrides": json.dumps(ov, sort_keys=True) if ov else "",
                "question_id": q.get("question_id", ""), "scored": scored, "lenient": lenient,
                "strict": strict, "first_attempt": int(lenient and q.get("total_iterations", 0) <= 1),
                "attempts": q.get("total_iterations", 0),
                "latency_ms": round(float(q.get("total_latency_ms", 0.0) or 0.0)),
                "final_sparql": q.get("final_sparql") or "",
            })
        roles[(role, model)] += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows from {sum(roles.values())} runs to {out}")
    for (role, model), n in sorted(roles.items()):
        print(f"  {role:14s} {model:26s} {n} runs")
    bad = [r for r in rows if r["ontology"] in NEVER]
    print(f"eNOVATION rows: {len(bad)} (must be 0)")


if __name__ == "__main__":
    main()
