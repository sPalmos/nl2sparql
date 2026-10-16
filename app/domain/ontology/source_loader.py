"""Load ontology sources into RDFLib graphs.

This module handles only source access and RDF parsing. It supports local
`.ttl`, `.owl`, and `.rdf` files plus existing SPARQL endpoints read through a
CONSTRUCT query. It does not write package files, resolve schemas, or build
runtime indexes.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import httpx
from rdflib import Graph

from app.domain.package import DomainError
from app.domain.ontology.graph_preparation import RDFLIB_FORMATS


SUPPORTED_SUFFIXES = {".ttl", ".owl", ".rdf"}


class OntologySourceError(DomainError):
    """Raised when an ontology source cannot be loaded."""


@dataclass(frozen=True)
class LoadedOntologySource:
    """RDF graph and source metadata loaded before graph preparation."""

    graph: Graph
    source_mode: str
    source_name: str
    source_path: Path | None
    content: bytes | None
    suffix: str | None
    query_endpoint: str | None


async def load_ontology_file(source: str | Path, *, source_filename: str | None = None) -> LoadedOntologySource:
    """Load a local ontology file into an RDF graph."""
    path = Path(source).resolve()
    if not path.exists():
        raise OntologySourceError(f"Ontology file not found: {path.as_posix()}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise OntologySourceError("Only .ttl, .owl, and .rdf files are supported")

    content = path.read_bytes()
    if not content:
        raise OntologySourceError("Ontology file is empty")

    graph = parse_rdf_content(content, suffix)
    return LoadedOntologySource(
        graph=graph,
        source_mode="file",
        source_name=source_filename or path.name,
        source_path=path,
        content=content,
        suffix=suffix,
        query_endpoint=None,
    )


async def load_sparql_endpoint(endpoint: str) -> LoadedOntologySource:
    """Load all triples from a SPARQL endpoint into an RDF graph."""
    graph = await graph_from_sparql_endpoint(endpoint)
    return LoadedOntologySource(
        graph=graph,
        source_mode="sparql_endpoint",
        source_name=endpoint,
        source_path=None,
        content=None,
        suffix=None,
        query_endpoint=endpoint,
    )


def parse_rdf_content(content: bytes, suffix: str) -> Graph:
    """Parse RDF bytes using the ontology file suffix."""
    graph = Graph()
    graph.parse(source=BytesIO(content), format=RDFLIB_FORMATS[suffix])
    return graph


async def graph_from_sparql_endpoint(endpoint: str) -> Graph:
    """Read an RDF graph from a SPARQL endpoint using a CONSTRUCT query."""
    query = "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }"
    headers = {
        "Accept": "text/turtle, application/rdf+xml, application/n-triples, text/plain",
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(endpoint, data={"query": query}, headers=headers)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise OntologySourceError(f"Failed to read triples from SPARQL endpoint: {endpoint}") from exc

    graph = Graph()
    parse_errors: list[Exception] = []
    for rdf_format in candidate_graph_formats(response.headers.get("content-type", "")):
        try:
            graph.parse(source=BytesIO(response.content), format=rdf_format)
            if len(graph) > 0:
                return graph
        except Exception as exc:  # pragma: no cover - fallback loop
            parse_errors.append(exc)

    raise OntologySourceError(
        f"Unable to parse RDF graph returned by SPARQL endpoint: {endpoint}"
    ) from (parse_errors[-1] if parse_errors else None)


def candidate_graph_formats(content_type: str) -> list[str]:
    """Return RDFLib parser formats to try for an endpoint response."""
    lowered = content_type.lower()
    if "text/turtle" in lowered:
        return ["turtle", "xml", "nt"]
    if "rdf+xml" in lowered or "application/xml" in lowered or "text/xml" in lowered:
        return ["xml", "turtle", "nt"]
    if "n-triples" in lowered or "plain" in lowered:
        return ["nt", "turtle", "xml"]
    return ["turtle", "xml", "nt"]
