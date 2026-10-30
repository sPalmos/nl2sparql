"""B2 summary: full schema with a 64k context (qwen3-coder:30b, eNOVATION), numbers only.

Run from the revision working copy:
    python analysis/b2_summary.py --paper-root <folder with the paper's runs>
Reads the run with TBox k = 1000 and logs/ollama_usage.jsonl (num_ctx 65536), and compares
with the paper's reference run of the same model. Prints no question text or IDs.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "analysis"))
import build_headline_tables as bht  # noqa: E402

MODEL = "qwen3-coder:30b"


def runs(root: Path):
    for path in sorted(root.rglob("results.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if data.get("results") and data.get("model_name") == MODEL \
                and bht.ontology_key(data.get("package_dir", "")) == "enovation" \
                and "heldout" not in (data.get("dataset_name", "") or "").lower():
            yield path, data


def per_question(data):
    out, aliases, prefixes = {}, None, None
    for q in data["results"]:
        if not q.get("gold_answers"):
            continue
        len_ = bool((q.get("comparison") or {}).get("exact_match"))
        st = False
        if len_ and q.get("final_answers") is not None:
            if aliases is None:
                aliases, prefixes = bht.package_context(data["package_dir"])
            st = bht.normalize_result_set(q["gold_answers"], prefixes, aliases) == \
                bht.normalize_result_set(q["final_answers"], prefixes, aliases)
        out[q["question_id"]] = (len_, st, q.get("total_iterations", 0) <= 1, q.get("total_latency_ms", 0.0))
    return out


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


ap = argparse.ArgumentParser()
ap.add_argument("--paper-root", default=r"..\NL-to-SPARQL-V4")
args = ap.parse_args()
full = [(d.get("timestamp", ""), d) for p, d in runs(Path("ontology_packages"))
        if (d.get("pipeline_config") or {}).get("retrieval_top_k") == 1000]
ref = [(d.get("timestamp", ""), d) for p, d in runs(Path(args.paper_root) / "ontology_packages")
       if bht.matches_reference(d.get("pipeline_config") or {})]
if not full or not ref:
    sys.exit(f"missing run (full schema: {len(full)}, reference: {len(ref)})")
fd, rd = max(full, key=lambda x: x[0])[1], max(ref, key=lambda x: x[0])[1]
F, R = per_question(fd), per_question(rd)
print(f"B2: {MODEL}, eNOVATION, full schema (TBox k = 1000), num_ctx 65536, vs the paper's reference run (k = 5, num_ctx 32768)\n")
for name, P in (("full schema, 64k", F), ("reference (paper)", R)):
    n = len(P)
    print(f"{name:20s} n {n}  EA lenient {100 * sum(v[0] for v in P.values()) / n:5.1f}  "
          f"strict {100 * sum(v[1] for v in P.values()) / n:5.1f}  "
          f"1st attempt {100 * sum(v[0] and v[2] for v in P.values()) / n:5.1f}  "
          f"latency {statistics.mean(v[3] for v in P.values()) / 1000:6.1f} s")
ids = sorted(set(F) & set(R))
for k, rule in ((0, "lenient"), (1, "strict")):
    b = sum(F[i][k] and not R[i][k] for i in ids)
    c = sum(R[i][k] and not F[i][k] for i in ids)
    print(f"{rule}: only full schema correct {b}, only reference correct {c}, exact McNemar p = {mcnemar(b, c):.3g}")
log = Path("logs") / "ollama_usage.jsonl"
if log.exists():
    calls = [json.loads(l) for l in open(log, encoding="utf-8")]
    calls = [c for c in calls if c.get("model") == MODEL and (c.get("options") or {}).get("num_ctx") == 65536]
    p = [c["prompt_eval_count"] for c in calls if c.get("prompt_eval_count") is not None]
    if p:
        print(f"\nOllama log: {len(calls)} calls; prompt_eval_count min {min(p)}, median {int(statistics.median(p))}, max {max(p)}; "
              f">= 65000: {sum(x >= 65000 for x in p)}; done_reason {dict(Counter(c.get('done_reason') for c in calls))}")
print(f"attempts recorded in the run: {sum(q.get('total_iterations', 0) for q in fd['results'])}")
