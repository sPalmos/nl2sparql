"""Per-ontology statistics table for the paper.

For each of the 20 reported ontologies it takes the package that produced the
main-table runs (same selection as build_headline_tables.py) and counts, on the
package's final graph:

  triples              all triples in the graph loaded into Fuseki
  classes              named classes: declared (owl:Class / rdfs:Class) or used
                       as the object of rdf:type, outside the W3C vocabularies
  object_properties    predicates outside the W3C vocabularies whose objects are IRIs
  datatype_properties  predicates outside the W3C vocabularies whose objects are literals
  instances            distinct subjects typed with one of those classes
  questions            evaluation questions in the dataset
  questions_scored     questions after excluding empty-gold ones (the paper's n)

Properties are counted as used in the data, so ontologies with and without
explicit OWL declarations are counted the same way. A predicate used with both
IRI and literal objects is counted in both columns (reported separately).

Usage (repository root):
    python ontology_stats.py
Writes analysis/table_ontology_stats.csv.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rdflib import Graph, Literal, URIRef                          # noqa: E402
from rdflib.namespace import OWL, RDF, RDFS                        # noqa: E402

from build_headline_tables import matches_reference, ontology_key  # noqa: E402

W3C = ("http://www.w3.org/1999/02/22-rdf-syntax-ns#", "http://www.w3.org/2000/01/rdf-schema#",
       "http://www.w3.org/2002/07/owl#", "http://www.w3.org/2001/XMLSchema#",
       "http://www.w3.org/2004/02/skos/core#")
RDF_SUFFIXES = (".ttl", ".nt", ".owl", ".rdf", ".n3", ".trig", ".nq")
SKIP_DIRS = {"indexes", "evaluation", "logs", "query-runs"}
MIN_QUESTIONS = 5


def is_w3c(term) -> bool:
    return isinstance(term, URIRef) and str(term).startswith(W3C)


def package_graph_file(package: Path) -> Path | None:
    """The package's own source graph (ontology/source.*).

    Falls back to the largest RDF file outside indexes/evaluation/logs only when
    no source file exists. Never picks ontology/schemas/*, the W3C vocabularies
    that onboarding downloads (for small Spider graphs, the downloaded OWL schema
    is larger than the source itself)."""
    sources = [p for p in (package / "ontology").glob("source.*") if p.suffix.lower() in RDF_SUFFIXES]
    if sources:
        return sources[0]
    candidates = [p for p in package.rglob("*")
                  if "schemas" not in p.relative_to(package).parts
                  if p.suffix.lower() in RDF_SUFFIXES
                  and not (SKIP_DIRS & set(p.relative_to(package).parts))]
    return max(candidates, key=lambda p: p.stat().st_size) if candidates else None


def stats(graph: Graph) -> dict:
    classes = {c for c in graph.subjects(RDF.type, OWL.Class) if isinstance(c, URIRef)}
    classes |= {c for c in graph.subjects(RDF.type, RDFS.Class) if isinstance(c, URIRef)}
    classes |= {o for o in graph.objects(None, RDF.type) if isinstance(o, URIRef)}
    classes = {c for c in classes if not is_w3c(c)}

    object_props, data_props = set(), set()
    for _, predicate, obj in graph:
        if is_w3c(predicate):
            continue
        if isinstance(obj, Literal):
            data_props.add(predicate)
        else:
            object_props.add(predicate)

    instances = {s for s, _, o in graph.triples((None, RDF.type, None))
                 if o in classes and s not in classes}
    return {
        "triples": len(graph),
        "classes": len(classes),
        "object_properties": len(object_props),
        "datatype_properties": len(data_props),
        "properties_both": len(object_props & data_props),
        "instances": len(instances),
    }


def main() -> None:
    root = Path(".").resolve()
    chosen: dict[str, tuple[str, Path]] = {}
    for path in sorted(root.rglob("results.json")):
        try:
            results = json.loads(path.read_text(encoding="utf-8"))
        except Exception:                                          # noqa: BLE001
            continue
        if not results.get("results") or not matches_reference(results.get("pipeline_config") or {}):
            continue
        ontology = ontology_key(results.get("package_dir", ""))
        stamp = results.get("timestamp", "")
        if ontology not in chosen or stamp > chosen[ontology][0]:
            chosen[ontology] = (stamp, path)

    rows = []
    for ontology, (_, path) in sorted(chosen.items()):
        results = json.loads(path.read_text(encoding="utf-8"))
        questions = [q for q in results["results"] if q.get("is_scored", True)]
        scored = [q for q in questions if q.get("gold_answers")]
        if len(scored) < MIN_QUESTIONS:
            continue
        package = Path(results["package_dir"])
        graph_file = package_graph_file(package)
        row = {"ontology": ontology, "package": package.name,
               "questions": len(questions), "questions_scored": len(scored)}
        if graph_file is None:
            row["note"] = "no RDF file found in package"
        else:
            graph = Graph()
            graph.parse(str(graph_file))
            row.update(stats(graph))
            row["graph_file"] = str(graph_file.relative_to(package))
        rows.append(row)
        print(f"  {ontology:38s} {row.get('triples', '-'):>9} triples  "
              f"{row.get('classes', '-'):>5} classes  {row['questions_scored']:>4} questions")

    fields = ["ontology", "triples", "classes", "object_properties", "datatype_properties",
              "properties_both", "instances", "questions", "questions_scored",
              "package", "graph_file", "note"]
    out = Path("analysis")
    out.mkdir(exist_ok=True)
    with (out / "table_ontology_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out / 'table_ontology_stats.csv'} ({len(rows)} ontologies)")


if __name__ == "__main__":
    main()