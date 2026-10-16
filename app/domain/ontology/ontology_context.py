"""Build the `ontology_context.json` payload from an RDF graph.

This module converts a prepared RDFLib graph into the runtime ontology context:
prefixes, classes, object/datatype properties, class hierarchy, and instance
statistics. It does not write the JSON file itself.
"""

from __future__ import annotations

from rdflib import BNode, Graph, Literal, RDF, RDFS, URIRef
from rdflib.namespace import OWL


def build_ontology_context(
    graph: Graph,
    *,
    ontology_name: str,
    source_filename: str,
) -> dict[str, object]:
    """Return the ontology context stored in `ontology_context.json`."""
    class_uris = _subjects_for_types(graph, {OWL.Class, RDFS.Class})
    object_property_uris = _subjects_for_types(graph, {OWL.ObjectProperty})
    datatype_property_uris = _subjects_for_types(graph, {OWL.DatatypeProperty})

    return {
        "ontology_name": ontology_name,
        "source_filename": source_filename,
        "triple_count": len(graph),
        "prefixes": _prefixes(graph),
        "classes": [_class_entry(graph, subject) for subject in class_uris],
        "object_properties": [
            _property_entry(graph, subject, "object_property") for subject in object_property_uris
        ],
        "datatype_properties": [
            _property_entry(graph, subject, "datatype_property") for subject in datatype_property_uris
        ],
        "class_hierarchy": _class_hierarchy(graph, class_uris),
        "instance_statistics": _instance_statistics(graph, class_uris),
        "naming_strategy": _naming_strategy(graph, class_uris),
    }


def _class_entry(graph: Graph, subject: URIRef) -> dict[str, object]:
    return {
        "uri": str(subject),
        "name": _local_name(subject),
        "label": _label_for(graph, subject),
        "comment": _comment_for(graph, subject),
        "parent_classes": sorted(
            str(parent)
            for parent in graph.objects(subject, RDFS.subClassOf)
            if isinstance(parent, URIRef)
        ),
    }


def _property_entry(graph: Graph, subject: URIRef, property_type: str) -> dict[str, object]:
    return {
        "uri": str(subject),
        "name": _local_name(subject),
        "label": _label_for(graph, subject),
        "comment": _comment_for(graph, subject),
        "domain": _property_schema_values(graph, subject, RDFS.domain),
        "range": _property_schema_values(graph, subject, RDFS.range),
        "inverse_properties": _inverse_properties(graph, subject),
        "property_type": property_type,
    }


def _inverse_properties(graph: Graph, subject: URIRef) -> list[str]:
    inverse_values: set[str] = set()
    for inverse in graph.objects(subject, OWL.inverseOf):
        if isinstance(inverse, URIRef):
            inverse_values.add(str(inverse))
    for inverse in graph.subjects(OWL.inverseOf, subject):
        if isinstance(inverse, URIRef):
            inverse_values.add(str(inverse))
    return sorted(inverse_values)


def _property_schema_values(graph: Graph, subject: URIRef, predicate: URIRef) -> list[str]:
    values: set[str] = set()
    for value in graph.objects(subject, predicate):
        values.update(_class_expression_values(graph, value))
    return sorted(values)


def _class_expression_values(graph: Graph, value: object) -> set[str]:
    if isinstance(value, URIRef):
        return {str(value)}

    if not isinstance(value, BNode):
        return set()

    union_head = graph.value(value, OWL.unionOf)
    if union_head is None:
        return set()

    return {
        str(item)
        for item in _rdf_list_items(graph, union_head)
        if isinstance(item, URIRef)
    }


def _rdf_list_items(graph: Graph, head: object) -> list[object]:
    items: list[object] = []
    current = head
    visited: set[object] = set()

    while current and current != RDF.nil and current not in visited:
        visited.add(current)
        first = graph.value(current, RDF.first)
        if first is not None:
            items.append(first)
        current = graph.value(current, RDF.rest)

    return items


def _class_hierarchy(graph: Graph, class_uris: list[URIRef]) -> list[dict[str, str]]:
    hierarchy = []
    for child in class_uris:
        for parent in graph.objects(child, RDFS.subClassOf):
            if isinstance(parent, URIRef):
                hierarchy.append({"parent": str(parent), "child": str(child)})
    return sorted(hierarchy, key=lambda item: (item["parent"], item["child"]))


def _instance_statistics(graph: Graph, class_uris: list[URIRef]) -> dict[str, object]:
    class_instances = []
    total_instances = 0
    for class_uri in class_uris:
        count = sum(1 for _ in graph.subjects(RDF.type, class_uri))
        if count:
            class_instances.append({"class_uri": str(class_uri), "count": count})
            total_instances += count
    return {
        "total_instances": total_instances,
        "class_instances": sorted(class_instances, key=lambda item: item["class_uri"]),
    }


# Datatype-property local names that commonly hold human-readable entity names,
# ordered by preference. Used only as a tiebreaker when a class has multiple
# candidate name properties in the actual data.
_NAME_PROPERTY_PREFERENCE = (
    "name", "title", "label", "preflabel",
    "first_name", "fname", "full_name", "fullname",
    "description",
)


def _naming_strategy(graph: Graph, class_uris: list[URIRef]) -> dict[str, object]:
    """Determine, from actual data, how each class exposes human-readable names.

    For each class that has instances, inspect the instances to see whether they
    carry rdfs:label. If they do, that class uses "rdfs:label". If they do not,
    look for the datatype property whose values best serve as the display name
    (a string-valued property, preferring conventional name properties).

    Returns:
    {
      "default": "rdfs:label" | "custom" | "mixed",
      "uses_rdfs_label": bool,        # true if ANY class uses rdfs:label
      "per_class": { "<ClassLocalName>": "rdfs:label" | "<propertyLocalName>" }
    }

    This is ground truth from the data, not a schema heuristic. It works per
    class, so mixed ontologies (some classes with rdfs:label, some with custom
    names) are handled correctly.
    """
    per_class: dict[str, str] = {}
    rdfs_label_count = 0
    custom_count = 0

    for class_uri in class_uris:
        instances = [s for s in graph.subjects(RDF.type, class_uri) if isinstance(s, URIRef)]
        if not instances:
            continue

        class_name = _local_name(class_uri)

        # Check whether instances of this class carry rdfs:label in the data
        has_rdfs_label = any(
            isinstance(graph.value(instance, RDFS.label), Literal)
            for instance in instances
        )

        if has_rdfs_label:
            per_class[class_name] = "rdfs:label"
            rdfs_label_count += 1
            continue

        # No rdfs:label — find the best string-valued datatype property to use
        name_property = _detect_name_property(graph, instances)
        if name_property:
            per_class[class_name] = name_property
            custom_count += 1

    # Decide the overall default
    if rdfs_label_count > 0 and custom_count == 0:
        default = "rdfs:label"
    elif custom_count > 0 and rdfs_label_count == 0:
        default = "custom"
    elif custom_count > 0 and rdfs_label_count > 0:
        default = "mixed"
    else:
        default = "rdfs:label"  # No instances found at all — safe default

    return {
        "default": default,
        "uses_rdfs_label": rdfs_label_count > 0,
        "per_class": per_class,
    }


def _detect_name_property(graph: Graph, instances: list[URIRef]) -> str | None:
    """Find the datatype property that best holds the display name for instances.

    Collects all string-valued predicates used by the sampled instances, then
    picks the one that best matches conventional name properties. Falls back to
    the most common string property if no conventional name is found.
    """
    # Sample up to 10 instances for efficiency
    sample = instances[:10]

    # Count string-valued predicates across the sample
    predicate_counts: dict[str, int] = {}
    for instance in sample:
        for predicate, obj in graph.predicate_objects(instance):
            if not isinstance(predicate, URIRef):
                continue
            if predicate in (RDF.type, RDFS.label, RDFS.comment):
                continue
            if isinstance(obj, Literal):
                # Prefer string literals (names are strings, not numbers/dates)
                datatype = getattr(obj, "datatype", None)
                is_stringy = datatype is None or "string" in str(datatype).lower()
                if is_stringy:
                    pred_local = _local_name(predicate)
                    predicate_counts[pred_local] = predicate_counts.get(pred_local, 0) + 1

    if not predicate_counts:
        return None

    # First, try conventional name properties in order of preference
    for preferred in _NAME_PROPERTY_PREFERENCE:
        for pred_local in predicate_counts:
            if pred_local.lower() == preferred:
                return pred_local

    # Then try any property containing "name"
    for pred_local in predicate_counts:
        if "name" in pred_local.lower() and "code" not in pred_local.lower():
            return pred_local

    # Fallback: the most frequently populated string property
    return max(predicate_counts.items(), key=lambda kv: kv[1])[0]


def _subjects_for_types(graph: Graph, rdf_types: set[URIRef]) -> list[URIRef]:
    subjects: set[URIRef] = set()
    for rdf_type in rdf_types:
        for subject in graph.subjects(RDF.type, rdf_type):
            if isinstance(subject, URIRef):
                subjects.add(subject)
    return sorted(subjects, key=str)


def _prefixes(graph: Graph) -> list[dict[str, str]]:
    used_namespaces = _used_namespaces(graph)
    prefixes: list[dict[str, str]] = []

    for prefix, namespace in graph.namespaces():
        namespace_text = str(namespace)
        if namespace_text not in used_namespaces:
            continue
        prefixes.append(
            {
                "prefix": ":" if prefix == "" else prefix,
                "namespace": namespace_text,
            }
        )

    return sorted(prefixes, key=lambda item: item["prefix"])


def _used_namespaces(graph: Graph) -> set[str]:
    namespaces: set[str] = set()
    for subject, predicate, object_value in graph:
        for value in (subject, predicate, object_value):
            if isinstance(value, URIRef):
                namespace = _namespace_for(value)
                if namespace:
                    namespaces.add(namespace)
    return namespaces


def _namespace_for(uri: URIRef) -> str:
    text = str(uri)
    if "#" in text:
        return text.rsplit("#", 1)[0] + "#"
    if "/" in text:
        return text.rsplit("/", 1)[0] + "/"
    return ""


def _label_for(graph: Graph, subject: URIRef) -> str:
    label = graph.value(subject, RDFS.label)
    if isinstance(label, Literal):
        return str(label)
    return _local_name(subject)


def _comment_for(graph: Graph, subject: URIRef) -> str | None:
    comment = graph.value(subject, RDFS.comment)
    if isinstance(comment, Literal):
        return str(comment)
    return None


def _local_name(subject: URIRef) -> str:
    text = str(subject)
    if "#" in text:
        return text.rsplit("#", 1)[-1]
    return text.rstrip("/").rsplit("/", 1)[-1]
