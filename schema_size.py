"""Size of the full class-based schema per ontology (for the full-schema baseline).
Usage: python schema_size.py   -> analysis/table_schema_size.csv"""
import csv, json
from pathlib import Path

rows = []
for package in sorted(Path("ontology_packages").glob("*-2026*")):
    chunks = package / "indexes" / "class_based" / "chunks.json"
    if not chunks.exists():
        continue
    data = json.loads(chunks.read_text(encoding="utf-8"))
    chars = sum(len(c.get("text") or "") for c in data)
    rows.append({"package": package.name, "chunks": len(data), "chars": chars,
                 "approx_tokens": chars // 4, "fits_32k_ctx": chars // 4 < 28000})
out = Path("analysis"); out.mkdir(exist_ok=True)
with (out / "table_schema_size.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    print(f"{r['package']:52s}{r['chunks']:>5d} chunks {r['approx_tokens']:>8d} tokens  fits={r['fits_32k_ctx']}")
