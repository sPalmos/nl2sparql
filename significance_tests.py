"""Paired per-question significance tests for the ablation study.

The pipeline is deterministic at temperature 0, so repeating a run gives the
same answers and adds no information. Significance therefore comes from pairing:
two configurations answer the same questions, and each question is either
solved by both, by neither, or by only one of them. Only the discordant
questions carry evidence.

For every comparison this script reports
  macro delta   mean over ontologies of the per-ontology EA difference
                (the number shown in the ablation table)
  micro delta   EA difference over all questions pooled
  b / c         questions solved only by the reference / only by the variant
  McNemar p     exact two-sided binomial test on b and c, pooled
  Holm p        McNemar p adjusted for multiple comparisons within each family
  95% CI        paired bootstrap on the macro delta, resampling questions
                within each ontology (stratified), 10,000 replicates

Scoring follows build_headline_tables.py: empty-gold questions are excluded,
lenient correctness is the evaluator's stored exact_match, and strict
correctness (with --strict) re-checks the lenient-correct questions with exact
row-set equality, aliases kept.

Usage (from the repository root, next to results_ablation.csv)
    python significance_tests.py
    python significance_tests.py --strict
    python significance_tests.py --skip spider-dog-kennels      # preview only
    python significance_tests.py --strict --substitute S4:spider-world-1=det_full_a

Outputs (in --out-dir, default analysis/)
    table_ablation_significance.csv     one row per comparison and rule
    table_ablation_per_ontology.csv     EA per configuration and ontology
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

REFERENCE = "S4"

# (family, reference, variant). The ladder compares consecutive steps; every
# other family compares a variant against the full system.
COMPARISONS = [
    ("ladder", "S0", "S1"),
    ("ladder", "S1", "S2"),
    ("ladder", "S2", "S3"),
    ("ladder", "S3", "S4"),
    ("ladder", "S0", "S4"),
    ("loo", REFERENCE, "loo_no_abox"),
    ("loo", REFERENCE, "loo_no_abox_rerank"),
    ("loo", REFERENCE, "loo_no_abox_gating"),
    ("loo", REFERENCE, "loo_no_correction"),
    ("loo", REFERENCE, "loo_no_adaptive_naming"),
    ("loo", REFERENCE, "loo_no_vocab_validation"),
    ("loo", REFERENCE, "loo_minimal_rules"),
    ("retrieval", REFERENCE, "chunk_property_based"),
    ("retrieval", REFERENCE, "chunk_composite"),
    ("retrieval", REFERENCE, "tbox_k3"),
    ("retrieval", REFERENCE, "tbox_k10"),
    ("retrieval", REFERENCE, "abox_k3"),
    ("retrieval", REFERENCE, "abox_k10"),
    ("retrieval", REFERENCE, "reactive_abox"),
    ("baseline", REFERENCE, "full_schema"),
    ("determinism", REFERENCE, "det_full_a"),
    ("determinism", REFERENCE, "det_full_b"),
    ("determinism", REFERENCE, "det_full_c"),
]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def latest_rows(csv_path: Path, skip: set[str]) -> dict[tuple[str, str], dict]:
    """Newest successful row per (config, ontology). A --redo appends a new row,
    so the newest one is the one that counts."""
    rows: dict[tuple[str, str], dict] = {}
    with csv_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok" or row["ontology"] in skip:
                continue
            key = (row["config"], row["ontology"])
            if key not in rows or row.get("timestamp", "") >= rows[key].get("timestamp", ""):
                rows[key] = row
    return rows


class StrictScorer:
    """Strict re-check using the evaluator's own normalisation and aliases."""

    def __init__(self) -> None:
        from evaluation.answer_comparison import normalize_result_set
        from evaluation.experiment_runner import (
            answer_aliases_from_package, prefix_map_from_package)
        self._normalize = normalize_result_set
        self._aliases_from = answer_aliases_from_package
        self._prefixes_from = prefix_map_from_package
        self._cache: dict[str, tuple] = {}

    def context(self, package_dir: str) -> tuple:
        if package_dir not in self._cache:
            package = Path(package_dir)
            try:
                aliases = self._aliases_from(package)
            except Exception:                                      # noqa: BLE001
                aliases = {}
            try:
                prefixes = self._prefixes_from(package)
            except Exception:                                      # noqa: BLE001
                prefixes = {}
            self._cache[package_dir] = (aliases, prefixes)
        return self._cache[package_dir]

    def correct(self, question: dict, package_dir: str) -> bool:
        generated = question.get("final_answers")
        if generated is None:
            return False
        aliases, prefixes = self.context(package_dir)
        return (self._normalize(question.get("gold_answers") or [], prefixes, aliases)
                == self._normalize(generated, prefixes, aliases))


def load_outcomes(row: dict, strict: StrictScorer | None) -> dict[str, dict[str, bool]]:
    """question_id -> {"lenient": bool, "strict": bool} for scored, non-empty-gold questions."""
    results = json.loads(Path(row["results_path"]).read_text(encoding="utf-8"))
    package_dir = results.get("package_dir", row.get("package_dir", ""))
    outcomes: dict[str, dict[str, bool]] = {}
    for question in results.get("results", []):
        if not question.get("is_scored", True):
            continue
        if not (question.get("gold_answers") or []):
            continue
        lenient = bool((question.get("comparison") or {}).get("exact_match"))
        entry = {"lenient": lenient}
        if strict is not None:
            # Strict implies lenient, so only lenient-correct answers need re-checking.
            entry["strict"] = lenient and strict.correct(question, package_dir)
        outcomes[question["question_id"]] = entry
    return outcomes


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar test: binomial on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def bootstrap_macro_ci(pairs_by_ontology: dict[str, list[tuple[int, int]]],
                       replicates: int, seed: int) -> tuple[float, float]:
    """Stratified paired bootstrap of the macro delta (in percentage points)."""
    rng = random.Random(seed)
    ontologies = list(pairs_by_ontology)
    # Per ontology, each question contributes (variant - reference) in {-1, 0, 1}.
    diffs = {o: [v - r for r, v in pairs_by_ontology[o]] for o in ontologies}
    samples = []
    for _ in range(replicates):
        total = 0.0
        for o in ontologies:
            d = diffs[o]
            total += sum(rng.choices(d, k=len(d))) / len(d)
        samples.append(100.0 * total / len(ontologies))
    samples.sort()
    low = samples[int(0.025 * replicates)]
    high = samples[int(0.975 * replicates) - 1]
    return low, high


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results_ablation.csv")
    parser.add_argument("--out-dir", default="analysis")
    parser.add_argument("--strict", action="store_true",
                        help="also test under the strict rule (needs the evaluation package)")
    parser.add_argument("--skip", nargs="*", default=[], help="ontologies to leave out")
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--substitute", nargs="*", default=[],
                        help="CONFIG:ONTOLOGY=SOURCE_CONFIG, e.g. S4:spider-world-1=det_full_a "
                             "(use that run as CONFIG for that ontology)")
    args = parser.parse_args()

    rows = latest_rows(Path(args.csv), set(args.skip))
    for spec in args.substitute:
        target, source = spec.split("=")
        config, ontology = target.split(":")
        if (source, ontology) not in rows:
            sys.exit(f"--substitute: no run for {source} on {ontology}")
        rows[(config, ontology)] = rows[(source, ontology)]
        print(f"Substituted: {config} on {ontology} now uses the {source} run "
              f"({rows[(source, ontology)].get('timestamp', '')}, EA {rows[(source, ontology)].get('accuracy', '')})")
    if not rows:
        sys.exit("No successful ablation rows found.")
    configs = sorted({c for c, _ in rows})
    ontologies = sorted({o for _, o in rows})

    strict = StrictScorer() if args.strict else None
    rules = ["lenient"] + (["strict"] if strict else [])

    print(f"Loading {len(rows)} runs ({len(configs)} configurations x "
          f"{len(ontologies)} ontologies)...")
    outcomes: dict[tuple[str, str], dict[str, dict[str, bool]]] = {}
    for key, row in sorted(rows.items()):
        outcomes[key] = load_outcomes(row, strict)

    # ---- per-ontology EA table, with a reconciliation against the CSV --------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_ontology_path = out_dir / "table_ablation_per_ontology.csv"
    mismatches = []
    with per_ontology_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["config", "ontology", "n", "ea_lenient", "ea_strict", "ea_csv"])
        for (config, ontology), result in sorted(outcomes.items()):
            n = len(result)
            lenient = round(100 * sum(q["lenient"] for q in result.values()) / n, 1) if n else ""
            strict_ea = (round(100 * sum(q["strict"] for q in result.values()) / n, 1)
                         if strict and n else "")
            csv_ea = rows[(config, ontology)].get("accuracy", "")
            writer.writerow([config, ontology, n, lenient, strict_ea, csv_ea])
            if csv_ea and lenient != "" and abs(float(csv_ea) - lenient) > 0.05:
                mismatches.append((config, ontology, csv_ea, lenient))
    if mismatches:
        print("\nNote: EA differs from results_ablation.csv where empty-gold questions "
              "were excluded here (the headline-table convention):")
        seen = set()
        for config, ontology, csv_ea, lenient in mismatches:
            if ontology not in seen:
                print(f"  {ontology}: e.g. {config} csv={csv_ea} here={lenient}")
                seen.add(ontology)

    # ---- paired tests -------------------------------------------------------
    records = []
    for family, ref, var in COMPARISONS:
        shared = [o for o in ontologies if (ref, o) in outcomes and (var, o) in outcomes]
        if not shared:
            continue
        for rule in rules:
            pairs_by_ontology: dict[str, list[tuple[int, int]]] = {}
            b = c = total = 0
            macro_terms = []
            per_ontology_bc = []
            for o in shared:
                ref_q, var_q = outcomes[(ref, o)], outcomes[(var, o)]
                ids = sorted(set(ref_q) & set(var_q))
                if set(ref_q) != set(var_q):
                    print(f"  ! {ref} vs {var} on {o}: question sets differ "
                          f"({len(ref_q)} vs {len(var_q)}); using {len(ids)} shared")
                pairs = [(int(ref_q[i][rule]), int(var_q[i][rule])) for i in ids]
                pairs_by_ontology[o] = pairs
                ob = sum(1 for r, v in pairs if r and not v)
                oc = sum(1 for r, v in pairs if v and not r)
                b, c, total = b + ob, c + oc, total + len(pairs)
                macro_terms.append(100.0 * (oc - ob) / len(pairs))
                per_ontology_bc.append(f"{o}:{ob}/{oc}")
            low, high = bootstrap_macro_ci(pairs_by_ontology, args.replicates, args.seed)
            records.append({
                "family": family, "reference": ref, "variant": var, "rule": rule,
                "ontologies": len(shared), "questions": total,
                "macro_delta_pp": round(sum(macro_terms) / len(macro_terms), 1),
                "micro_delta_pp": round(100.0 * (c - b) / total, 1),
                "ref_only_b": b, "variant_only_c": c,
                "mcnemar_p": mcnemar_exact(b, c),
                "ci95_low_pp": round(low, 1), "ci95_high_pp": round(high, 1),
                "per_ontology_b/c": " ".join(per_ontology_bc),
            })

    # Holm within each (family, rule); determinism is a check, not a hypothesis.
    for family in {r["family"] for r in records}:
        for rule in rules:
            group = [r for r in records if r["family"] == family and r["rule"] == rule]
            for record, adjusted in zip(group, holm([r["mcnemar_p"] for r in group])):
                record["holm_p"] = adjusted

    significance_path = out_dir / "table_ablation_significance.csv"
    fields = ["family", "reference", "variant", "rule", "ontologies", "questions",
              "macro_delta_pp", "micro_delta_pp", "ref_only_b", "variant_only_c",
              "mcnemar_p", "holm_p", "ci95_low_pp", "ci95_high_pp", "per_ontology_b/c"]
    with significance_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({**record,
                             "mcnemar_p": f"{record['mcnemar_p']:.4g}",
                             "holm_p": f"{record['holm_p']:.4g}"})

    # ---- console summary ----------------------------------------------------
    for rule in rules:
        print(f"\n{'=' * 104}\nPAIRED TESTS — {rule} rule  "
              f"(delta = variant minus reference, percentage points)\n{'=' * 104}")
        print(f"{'comparison':36s}{'macro':>8s}{'micro':>8s}{'b':>5s}{'c':>5s}"
              f"{'p':>10s}{'Holm p':>10s}   95% CI (macro)")
        print("-" * 104)
        family = None
        for r in [r for r in records if r["rule"] == rule]:
            if r["family"] != family:
                family = r["family"]
                print(f"[{family}]")
            mark = "*" if r["holm_p"] < 0.05 else " "
            print(f"  {r['reference'] + ' -> ' + r['variant']:34s}"
                  f"{r['macro_delta_pp']:>+8.1f}{r['micro_delta_pp']:>+8.1f}"
                  f"{r['ref_only_b']:>5d}{r['variant_only_c']:>5d}"
                  f"{r['mcnemar_p']:>10.3g}{r['holm_p']:>10.3g}{mark}  "
                  f"[{r['ci95_low_pp']:+.1f}, {r['ci95_high_pp']:+.1f}]")
    print("\n* Holm-adjusted p < 0.05 within its family.  b = solved only by the "
          "reference, c = solved only by the variant.")
    print(f"\nWrote {significance_path}\n      {per_ontology_path}")


if __name__ == "__main__":
    main()