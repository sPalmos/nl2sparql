"""Build instance-level ABox chunks for retrieval."""

from __future__ import annotations

from rdflib import Graph, Literal, RDF, RDFS, URIRef
from rdflib.namespace import OWL, SKOS


_TYPE_EXCLUSIONS = {
    OWL.Class,
    RDFS.Class,
    OWL.ObjectProperty,
    OWL.DatatypeProperty,
    OWL.AnnotationProperty,
    OWL.Ontology,
}

_LITERAL_EXCLUSIONS = {RDFS.comment}
_OBJECT_EXCLUSIONS = {RDF.type, RDFS.label, RDFS.comment, SKOS.prefLabel}
_NAME_KEYWORDS = (
    "label",
    "name",
    "title",
    "identifier",
    "identification",
    "code",
    "id",
)


def build_abox_chunks(
    graph: Graph,
    ontology_context: dict[str, object],
    *,
    max_literals_per_instance: int = 12,
    max_facts_per_instance: int = 24,
) -> list[dict[str, object]]:
    """Return one retrieval chunk per known ontology instance.

    Instances are subjects with rdf:type pointing at a class recorded in
    ontology_context. The chunk text is intentionally URI-forward: retrieval can
    match flexible names/literals, while generation can ground to exact URIs.
    """
    known_class_uris = _known_class_uris(ontology_context)
    class_names = _class_names_by_uri(ontology_context)
    chunks: list[dict[str, object]] = []

    for instance_uri in _typed_instances(graph, known_class_uris):
        types = [
            class_names.get(str(class_uri), _local_name(str(class_uri)))
            for class_uri in graph.objects(instance_uri, RDF.type)
            if isinstance(class_uri, URIRef) and str(class_uri) in known_class_uris
        ]
        if not types:
            continue

        literals = _literal_facts(graph, instance_uri, max_literals_per_instance)
        names = _name_values(literals)
        object_facts = _object_facts(graph, instance_uri, max_facts_per_instance)
        display_name = _display_name(instance_uri, names)
        text = _chunk_text(
            uri=str(instance_uri),
            local_name=_local_name(str(instance_uri)),
            display_name=display_name,
            types=types,
            names=names,
            literal_facts=literals,
            object_facts=object_facts,
        )

        chunks.append(
            {
                "chunk_type": "abox_instance",
                "uri": str(instance_uri),
                "local_name": _local_name(str(instance_uri)),
                "display_name": display_name,
                "types": sorted(set(types)),
                "names": names,
                "literal_facts": literals,
                "facts": object_facts,
                "text": text,
                "metadata": {
                    "uri": str(instance_uri),
                    "types": sorted(set(types)),
                    "names": names,
                },
            }
        )

    return sorted(chunks, key=lambda item: str(item.get("uri", "")))


def _known_class_uris(ontology_context: dict[str, object]) -> set[str]:
    classes = ontology_context.get("classes", [])
    if not isinstance(classes, list):
        return set()
    return {
        str(item["uri"])
        for item in classes
        if isinstance(item, dict) and isinstance(item.get("uri"), str)
        and str(item["uri"]) != str(OWL.NamedIndividual)
    }


def _class_names_by_uri(ontology_context: dict[str, object]) -> dict[str, str]:
    classes = ontology_context.get("classes", [])
    if not isinstance(classes, list):
        return {}
    result: dict[str, str] = {}
    for item in classes:
        if not isinstance(item, dict):
            continue
        uri = item.get("uri")
        name = item.get("name") or item.get("label") or uri
        if isinstance(uri, str) and name:
            result[uri] = _local_name(str(name))
    return result


def _typed_instances(graph: Graph, known_class_uris: set[str]) -> list[URIRef]:
    instances: set[URIRef] = set()
    for subject, rdf_type in graph.subject_objects(RDF.type):
        if not isinstance(subject, URIRef) or not isinstance(rdf_type, URIRef):
            continue
        if rdf_type in _TYPE_EXCLUSIONS:
            continue
        if str(rdf_type) in known_class_uris:
            instances.add(subject)
    return sorted(instances, key=str)


def _literal_facts(
    graph: Graph,
    subject: URIRef,
    limit: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen = set()
    for predicate, obj in graph.predicate_objects(subject):
        if not isinstance(predicate, URIRef) or not isinstance(obj, Literal):
            continue
        if predicate in _LITERAL_EXCLUSIONS:
            continue
        value = str(obj).strip()
        if not value:
            continue
        key = (str(predicate), value)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "property": _compact_uri(str(predicate)),
                "property_uri": str(predicate),
                "value": value,
            }
        )
    return sorted(rows, key=_literal_sort_key)[:limit]


def _name_values(literals: list[dict[str, str]]) -> list[dict[str, str]]:
    names = [
        item
        for item in literals
        if any(keyword in item["property"].lower() for keyword in _NAME_KEYWORDS)
    ]
    return sorted(names, key=_literal_sort_key)[:8]


def _object_facts(
    graph: Graph,
    subject: URIRef,
    limit: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen = set()
    for predicate, obj in graph.predicate_objects(subject):
        if not isinstance(predicate, URIRef) or not isinstance(obj, URIRef):
            continue
        if predicate in _OBJECT_EXCLUSIONS:
            continue
        key = (str(predicate), str(obj))
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "predicate": _compact_uri(str(predicate)),
                "predicate_uri": str(predicate),
                "object_uri": str(obj),
                "object_label": _resource_label(graph, obj),
                "object_local_name": _local_name(str(obj)),
            }
        )
    return sorted(rows, key=lambda item: (item["predicate"], item["object_label"]))[:limit]


def _resource_label(graph: Graph, uri: URIRef) -> str:
    for predicate in (RDFS.label, SKOS.prefLabel):
        value = graph.value(uri, predicate)
        if isinstance(value, Literal) and str(value).strip():
            return str(value).strip()
    return _local_name(str(uri))


def _display_name(instance_uri: URIRef, names: list[dict[str, str]]) -> str:
    if names:
        return names[0]["value"]
    return _local_name(str(instance_uri))


def _chunk_text(
    *,
    uri: str,
    local_name: str,
    display_name: str,
    types: list[str],
    names: list[dict[str, str]],
    literal_facts: list[dict[str, str]],
    object_facts: list[dict[str, str]],
) -> str:
    name_text = _format_name_values(names)
    literal_text = _format_literal_facts(literal_facts)
    object_text = _format_object_facts(object_facts)
    return (
        f"Instance: {display_name}\n\n"
        f"URI: <{uri}>\n\n"
        f"Local name: {local_name}\n\n"
        f"Types: {', '.join(sorted(set(types)))}\n\n"
        f"Name or identifier values:\n{name_text}\n\n"
        f"Datatype facts:\n{literal_text}\n\n"
        f"Outgoing object facts:\n{object_text}"
    )


def _format_name_values(values: list[dict[str, str]]) -> str:
    if not values:
        return "- None detected"
    return "\n".join(f"- {item['property']}: {item['value']}" for item in values)


def _format_literal_facts(values: list[dict[str, str]]) -> str:
    if not values:
        return "- None"
    return "\n".join(f"- {item['property']}: {item['value']}" for item in values)


def _format_object_facts(values: list[dict[str, str]]) -> str:
    if not values:
        return "- None"
    return "\n".join(
        f"- {item['predicate']} -> {item['object_label']} (<{item['object_uri']}>)"
        for item in values
    )


def _literal_sort_key(item: dict[str, str]) -> tuple[int, str, str]:
    property_name = item["property"].lower()
    is_name = any(keyword in property_name for keyword in _NAME_KEYWORDS)
    return (0 if is_name else 1, property_name, item["value"].lower())


def _compact_uri(value: str) -> str:
    return _local_name(value)


def _local_name(value: str) -> str:
    if "#" in value:
        return value.rsplit("#", 1)[-1]
    if "/" in value:
        return value.rstrip("/").rsplit("/", 1)[-1]
    if ":" in value:
        return value.rsplit(":", 1)[-1]
    return value
