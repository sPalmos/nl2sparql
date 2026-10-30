"""Compare two runs question by question (neutrality check for paper-v1.1).

Usage:
    python compare_runs.py <run A dir or results.json> <run B dir or results.json>
Identical means: same final SPARQL, same final answers, same number of attempts,
same per-attempt SPARQL and execution status, same comparison (scores).
"""
import json
import sys
from pathlib import Path


def load(arg):
    p = Path(arg)
    if p.is_dir():
        p = p / "results.json"
    return {q["question_id"]: q for q in json.load(open(p, encoding="utf-8"))["results"]}


def key(q):
    its = [(it.get("generated_sparql"), it.get("execution_status")) for it in q.get("iterations") or []]
    return (q.get("final_sparql"), json.dumps(q.get("final_answers"), sort_keys=True),
            q.get("total_iterations"), its, json.dumps(q.get("comparison"), sort_keys=True))


a, b = load(sys.argv[1]), load(sys.argv[2])
common = sorted(set(a) & set(b))
diff = [qid for qid in common if key(a[qid]) != key(b[qid])]
print(f"questions A {len(a)}, B {len(b)}, common {len(common)}, identical {len(common) - len(diff)}, different {len(diff)}")
for qid in diff:
    print("DIFFERENT", qid)
