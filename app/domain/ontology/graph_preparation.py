"""Prepare the `FinalGraph` used by ontology onboarding.

This module analyzes a loaded RDF graph, classifies it as schema-only, mixed, or
instances-only, checks whether instance classes are declared, optionally resolves
missing schema namespaces, and merges resolved schema triples into the final RDF
graph. It does not read source files or write package artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import httpx
from rdflib import Graph, RDF, RDFS, URIRef
from rdflib.namespace import OWL


RDFLIB_FORMATS = {
    ".ttl": "turtle",
    ".owl": "xml",
    ".rdf": "xml",
}

SCHEMA_TYPE_URIS = {
    str(OWL.Class),
    str(RDFS.Class),
    str(OWL.ObjectProperty),
    str(OWL.DatatypeProperty),
    str(OWL.AnnotationProperty),
    str(OWL.Ontology),
    str(RDF.Property),
}


@dataclass(frozen=True)
class DetectionResult:
    """Lightweight graph statistics used for mode classification."""

    classes_count: int
    properties_count: int
    instances_count: int


@dataclass(frozen=True)
class CoverageResult:
    """Describe whether instance class URIs are declared in the uploaded graph."""

    status: str
    instance_type_uris: list[str]
    declared_class_uris: list[str]
    missing_class_uris: list[str]
    missing_namespaces: list[str]


@dataclass(frozen=True)
class ResolvedSchemaFile:
    """One externally resolved schema file stored in the ontology package."""

    source_namespace: str
    url: str
    filename: str
    content: bytes
    suffix: str


@dataclass(frozen=True)
class ResolvedSchemas:
    """Schemas discovered while preparing the final graph."""

    resolved_files: list[ResolvedSchemaFile]
    attempted_urls: list[str]
    failed_urls: list[str]


@dataclass(frozen=True)
class FinalGraph:
    """Prepared ontology graph plus the analysis that produced it."""

    initial_graph: Graph
    graph: Graph
    mode: str
    detection: DetectionResult
    coverage: CoverageResult
    resolved_schemas: ResolvedSchemas


async def prepare_final_graph(initial_graph: Graph, *, resolve_missing_schemas: bool = True) -> FinalGraph:
    """Analyze and enrich one loaded RDF graph."""
    detection = detect_graph(initial_graph)
    mode = classify_mode(detection)
    coverage = analyze_schema_coverage(initial_graph)
    resolved_schemas = (
        await resolve_schemas_for_namespaces(coverage.missing_namespaces)
        if resolve_missing_schemas
        else ResolvedSchemas(resolved_files=[], attempted_urls=[], failed_urls=[])
    )
    graph = build_final_graph(initial_graph, resolved_schemas.resolved_files)
    return FinalGraph(
        initial_graph=initial_graph,
        graph=graph,
        mode=mode,
        detection=detection,
        coverage=coverage,
        resolved_schemas=resolved_schemas,
    )


def detect_graph(graph: Graph) -> DetectionResult:
    """Run a fast graph scan used to classify the ontology file."""
    class_uris = _subjects_for_types(graph, {OWL.Class, RDFS.Class})
    object_property_uris = _subjects_for_types(graph, {OWL.ObjectProperty})
    datatype_property_uris = _subjects_for_types(graph, {OWL.DatatypeProperty})
    instances = {str(subject) for subject in _instance_subjects(graph)}

    return DetectionResult(
        classes_count=len(class_uris),
        properties_count=len(object_property_uris) + len(datatype_property_uris),
        instances_count=len(instances),
    )


def classify_mode(detection: DetectionResult) -> str:
    """Map fast graph counts to one of the supported ontology modes."""
    if detection.classes_count > 0 and detection.instances_count == 0:
        return "schema-only"
    if detection.classes_count > 0 and detection.instances_count > 0:
        return "mixed"
    if detection.classes_count == 0 and detection.instances_count > 0:
        return "instances-only"
    return "schema-only"


def analyze_schema_coverage(graph: Graph) -> CoverageResult:
    """Compare instance rdf:type class URIs with class declarations."""
    instance_type_uris = _instance_type_uris(graph)
    declared_class_uris = _subjects_for_types(graph, {OWL.Class, RDFS.Class})
    declared_class_uri_strings = {str(class_uri) for class_uri in declared_class_uris}

    missing_class_uris = sorted(
        str(class_uri)
        for class_uri in instance_type_uris
        if str(class_uri) not in declared_class_uri_strings
    )
    missing_namespaces = sorted(
        namespace
        for namespace in {_namespace_of(URIRef(class_uri)) for class_uri in missing_class_uris}
        if namespace
    )

    return CoverageResult(
        status="complete" if not missing_class_uris else "incomplete",
        instance_type_uris=sorted(str(class_uri) for class_uri in instance_type_uris),
        declared_class_uris=sorted(str(class_uri) for class_uri in declared_class_uris),
        missing_class_uris=missing_class_uris,
        missing_namespaces=missing_namespaces,
    )


async def resolve_schemas_for_namespaces(namespaces: list[str]) -> ResolvedSchemas:
    """Try to download RDF schemas for the provided missing class namespaces."""
    attempted_urls: list[str] = []
    failed_urls: list[str] = []
    resolved_files: list[ResolvedSchemaFile] = []
    seen_urls: set[str] = set()

    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        for namespace in namespaces:
            for url in _candidate_schema_urls(namespace):
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                attempted_urls.append(url)
                try:
                    response = await client.get(url)
                except httpx.HTTPError:
                    continue
                if response.status_code >= 400:
                    continue

                suffix = _suffix_from_url_or_type(url, response.headers.get("content-type", ""))
                if suffix not in RDFLIB_FORMATS:
                    continue

                try:
                    schema_graph = Graph()
                    schema_graph.parse(source=BytesIO(response.content), format=RDFLIB_FORMATS[suffix])
                except Exception:
                    failed_urls.append(url)
                    continue

                if len(schema_graph) == 0:
                    failed_urls.append(url)
                    continue

                filename = f"schema-{len(resolved_files) + 1}{suffix}"
                resolved_files.append(
                    ResolvedSchemaFile(
                        source_namespace=namespace,
                        url=url,
                        filename=filename,
                        content=response.content,
                        suffix=suffix,
                    )
                )
                break

    return ResolvedSchemas(
        resolved_files=resolved_files,
        attempted_urls=attempted_urls,
        failed_urls=failed_urls,
    )


def build_final_graph(original_graph: Graph, schema_files: list[ResolvedSchemaFile]) -> Graph:
    """Combine the uploaded ontology graph with any resolved schemas."""
    final_graph = Graph()
    _copy_namespaces(final_graph, original_graph)
    for triple in original_graph:
        final_graph.add(triple)

    for schema_file in schema_files:
        schema_graph = Graph()
        schema_graph.parse(source=BytesIO(schema_file.content), format=RDFLIB_FORMATS[schema_file.suffix])
        _copy_namespaces(final_graph, schema_graph)
        for triple in schema_graph:
            final_graph.add(triple)

    return final_graph


def _copy_namespaces(target_graph: Graph, source_graph: Graph) -> None:
    for prefix, namespace in source_graph.namespaces():
        target_graph.bind(prefix, namespace)


def _instance_subjects(graph: Graph) -> set[URIRef]:
    instances: set[URIRef] = set()
    for subject, rdf_type in graph.subject_objects(RDF.type):
        if not isinstance(subject, URIRef) or not isinstance(rdf_type, URIRef):
            continue
        if str(rdf_type) in SCHEMA_TYPE_URIS:
            continue
        instances.add(subject)
    return instances


def _instance_type_uris(graph: Graph) -> list[URIRef]:
    instance_subjects = _instance_subjects(graph)
    instance_type_uris: set[URIRef] = set()
    for subject in instance_subjects:
        for rdf_type in graph.objects(subject, RDF.type):
            if not isinstance(rdf_type, URIRef):
                continue
            if str(rdf_type) in SCHEMA_TYPE_URIS:
                continue
            instance_type_uris.add(rdf_type)
    return sorted(instance_type_uris, key=str)


def _candidate_schema_urls(namespace: str) -> list[str]:
    base = namespace.rstrip("#/")
    return [namespace, base, f"{base}.owl", f"{base}.rdf", f"{base}.ttl"]


def _suffix_from_url_or_type(url: str, content_type: str) -> str | None:
    lowered_type = content_type.lower()
    lowered_url = url.lower()
    if "text/turtle" in lowered_type or lowered_url.endswith(".ttl"):
        return ".ttl"
    if "rdf+xml" in lowered_type or lowered_url.endswith(".owl") or lowered_url.endswith(".rdf"):
        return ".owl" if lowered_url.endswith(".owl") else ".rdf"
    return None


def _namespace_of(subject: URIRef) -> str | None:
    text = str(subject)
    if "#" in text:
        head, _, _ = text.rpartition("#")
        return f"{head}#"
    if "/" in text:
        head, _, _ = text.rpartition("/")
        return f"{head}/"
    return None


def _subjects_for_types(graph: Graph, rdf_types: set[URIRef]) -> list[URIRef]:
    subjects: set[URIRef] = set()
    for rdf_type in rdf_types:
        for subject in graph.subjects(RDF.type, rdf_type):
            if isinstance(subject, URIRef):
                subjects.add(subject)
    return sorted(subjects, key=str)
