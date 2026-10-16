"""Compare the regenerated tables with the paper's tables.

    python analysis/compare_tables.py                       # output/tables vs results/tables
    python analysis/compare_tables.py --expected <folder>   # against another folder

Rows are compared as exact text, independent of order. Package folder names carry
an onboarding timestamp (e.g. bestiary-20260923-1041); when two tables differ only
in those timestamps, the table is reported as identical apart from package names.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TABLES = [
    "table_main_matrix.csv", "table_model_summary.csv", "table_sensitivity.csv",
    "table_ablation_per_ontology.csv", "table_ablation_significance.csv",
    "table_api_per_ontology.csv", "table_api_summary.csv", "table_api_significance.csv",
    "table_error_categories.csv", "table_error_flags.csv", "table_difficulty_levels.csv",
    "table_ontology_stats.csv", "table_prompt_length.csv", "table_schema_size.csv",
]
STAMP = re.compile(r"-\d{8}-\d{4}(?:-\d+)?$")


def read(path: Path) -> tuple[list[str], list[tuple[str, ...]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [tuple(c.strip() for c in r) for r in csv.reader(handle) if r]
    return list(rows[0]), rows[1:]


def without_stamps(header: list[str], rows: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
    if "package" not in header:
        return rows
    col = header.index("package")
    return [r[:col] + (STAMP.sub("", r[col]),) + r[col + 1:] for r in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", default=str(ROOT / "results" / "tables"))
    parser.add_argument("--actual", default=str(ROOT / "output" / "tables"))
    args = parser.parse_args()
    expected_dir, actual_dir = Path(args.expected), Path(args.actual)

    failures = 0
    for name in TABLES:
        exp_path, act_path = expected_dir / name, actual_dir / name
        if not exp_path.exists() or not act_path.exists():
            print(f"MISSING    {name}  (expected: {exp_path.exists()}, regenerated: {act_path.exists()})")
            failures += 1
            continue
        exp_head, exp = read(exp_path)
        act_head, act = read(act_path)
        if exp_head != act_head:
            print(f"DIFFERENT  {name}: header changed\n    expected: {exp_head}\n    now:      {act_head}")
            failures += 1
            continue
        if Counter(exp) == Counter(act):
            print(f"IDENTICAL  {name}")
            continue
        if Counter(without_stamps(exp_head, exp)) == Counter(without_stamps(act_head, act)):
            print(f"IDENTICAL  {name}  (apart from package timestamps)")
            continue
        failures += 1
        missing, extra = Counter(exp) - Counter(act), Counter(act) - Counter(exp)
        print(f"DIFFERENT  {name}: {sum(missing.values())} expected rows missing, {sum(extra.values())} new rows")
        for r in list(missing)[:5]:
            print(f"    expected: {','.join(r)}")
        for r in list(extra)[:5]:
            print(f"    now:      {','.join(r)}")
    print("\nALL TABLES IDENTICAL" if not failures else f"\n{failures} table(s) differ")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
