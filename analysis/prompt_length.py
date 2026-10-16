"""Prompt length of S4 versus full_schema on the 6 ablation ontologies.

Reads the readable trace of every question in the newest S4 and full_schema
runs (paths come from results_ablation.csv -> results.json ->
readable_trace_path) and measures the GENERATION PROMPT section, i.e. the
first-attempt prompt as rendered, before any truncation by the runtime.

Tokens are estimated as characters / 4 (the runs do not record token counts).
If the `transformers` package and the Qwen2.5 tokenizer are available locally,
add --tokenizer Qwen/Qwen2.5-Coder-32B-Instruct for exact counts.

Usage:  python analysis/prompt_length.py [--configs S4 full_schema] [--tokenizer NAME]
Writes output/tables/table_prompt_length.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def newest_rows(configs: list[str]) -> dict[tuple[str, str], dict]:
    rows: dict[tuple[str, str], dict] = {}
    with (ROOT / "output" / "results_ablation.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["status"] != "ok" or row["config"] not in configs:
                continue
            key = (row["config"], row["ontology"])
            if key not in rows or row["timestamp"] >= rows[key]["timestamp"]:
                rows[key] = row
    return rows


def prompt_text(trace: Path) -> str | None:
    if not trace.exists():
        return None
    text = trace.read_text(encoding="utf-8", errors="replace")
    start, end = text.find("GENERATION PROMPT"), text.find("RETRIEVED ABOX CONTEXT")
    if start < 0:
        return None
    return text[start + len("GENERATION PROMPT"): end if end > start else None].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", default=["S4", "full_schema"])
    parser.add_argument("--tokenizer", default="")
    args = parser.parse_args()

    count = None
    if args.tokenizer:
        from transformers import AutoTokenizer                           # noqa: PLC0415
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        count = lambda s: len(tok.encode(s))                              # noqa: E731

    out_rows = []
    for (config, ontology), row in sorted(newest_rows(args.configs).items(), key=lambda kv: (kv[0][1], kv[0][0])):
        results = json.loads(Path(row["results_path"]).read_text(encoding="utf-8"))
        chars, tokens, missing = [], [], 0
        for q in results.get("results", []):
            text = prompt_text(Path(q.get("readable_trace_path") or ""))
            if text is None:
                missing += 1
                continue
            chars.append(len(text))
            tokens.append(count(text) if count else len(text) / 4)
        if not tokens:
            print(f"  {ontology} {config}: no traces found")
            continue
        out_rows.append({
            "ontology": ontology, "config": config, "questions": len(tokens), "missing_traces": missing,
            "mean_tokens": round(statistics.mean(tokens)), "median_tokens": round(statistics.median(tokens)),
            "max_tokens": round(max(tokens)), "token_method": "tokenizer" if count else "chars/4",
        })
        print(f"  {ontology:38s}{config:13s} mean {out_rows[-1]['mean_tokens']:>7} "
              f"median {out_rows[-1]['median_tokens']:>7} max {out_rows[-1]['max_tokens']:>7}"
              + (f"  ({missing} traces missing)" if missing else ""))

    (ROOT / "output" / "tables").mkdir(parents=True, exist_ok=True)
    with (ROOT / "output" / "tables" / "table_prompt_length.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)
    print("\nwrote output/tables/table_prompt_length.csv")


if __name__ == "__main__":
    main()
