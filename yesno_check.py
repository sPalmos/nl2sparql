"""Where the yes/no (question_type = boolean) questions come from, and their EA.

Same question set and runs as the main table (scored, non-empty gold, 20
ontologies, newest reference run per ontology x model). Prints the number of
boolean questions per ontology and the pooled lenient EA of each model on them.

Usage (repository root):  python yesno_check.py
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from label_breakdown import dataset_labels  # noqa: E402
from difficulty_levels import main_table_runs  # noqa: E402
from build_headline_tables import MODEL_ORDER  # noqa: E402

labels = dataset_labels()
runs = main_table_runs()
per_ontology = Counter()
hits, counts = defaultdict(int), defaultdict(int)
seen = set()
for (ontology, model), path in sorted(runs.items()):
    results = json.loads(path.read_text(encoding="utf-8"))
    qlabels = labels.get(results.get("dataset_name", ""), {})
    for q in results["results"]:
        if not q.get("is_scored", True) or not (q.get("gold_answers") or []):
            continue
        if qlabels.get(str(q.get("question_id")), {}).get("question_type") != "boolean":
            continue
        if (ontology, q.get("question_id")) not in seen:
            seen.add((ontology, q.get("question_id")))
            per_ontology[ontology] += 1
        counts[model] += 1
        hits[model] += bool((q.get("comparison") or {}).get("exact_match"))

print("Boolean questions per ontology (scored):")
for ontology, n in sorted(per_ontology.items()):
    print(f"  {ontology:38s} {n}")
print(f"  {'TOTAL':38s} {sum(per_ontology.values())}")
print("\nPooled lenient EA on boolean questions:")
for m in MODEL_ORDER:
    if counts[m]:
        print(f"  {m:20s} {100 * hits[m] / counts[m]:5.1f}%  ({hits[m]}/{counts[m]})")
