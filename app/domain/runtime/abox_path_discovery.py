"""Discover actual data paths from the ABox to guide SPARQL correction.

When a generated query is syntactically valid but returns zero results, this
module queries the SPARQL endpoint to find what properties actually connect
the classes mentioned in the query. The discovered paths are formatted as a
correction hint that tells the LLM which properties genuinely have data.

This inspects actual instance data (the ABox), not just the schema, to explain
why a query returned nothing and what the correct traversal would be.
"""

from __future__ import annotations

import re

import httpx

from app.core.config import settings


async def discover_paths_for_correction(
    endpoint_url: str,
    failed_query: str,
    ontology_context: dict[str, object],
) -> str:
    """Discover actual ABox paths and label values to guide correction.

    Extracts class types from the failed query, then for each class discovers:
    (1) the object properties that actually connect its instances (structural
    hints), and (2) a sample of the actual name/label values its instances carry
    (value hints). The value hints fix the common failure where the query's
    FILTER string does not match the real label in the data (e.g. searching for
    "eNOVATION network" when the instance is labeled just "eNOVATION").

    Returns a formatted hint string, or empty string on any failure (graceful).
    """
    if not endpoint_url or not failed_query:
        return ""

    try:
        prefix_block = _build_prefix_block(ontology_context)
        class_uris = _extract_class_uris(failed_query, ontology_context)
        if not class_uris:
            return ""

        naming = ontology_context.get("naming_strategy", {})
        per_class = naming.get("per_class", {}) if isinstance(naming, dict) else {}

        path_hints: list[str] = []
        value_hints: list[str] = []

        for class_uri in class_uris:
            class_name = _local_name(class_uri)

            # (1) Structural: what properties connect instances of this class
            outgoing = await _discover_outgoing_properties(endpoint_url, class_uri, prefix_block)
            if outgoing:
                props_text = "; ".join(
                    f":{p['property']} -> {p['target_class']}" for p in outgoing
                )
                path_hints.append(f"{class_name} instances connect via: {props_text}")

            # (2) Values: the actual name/label values instances carry
            name_property = per_class.get(class_name, "rdfs:label")
            labels = await _discover_instance_labels(
                endpoint_url, class_uri, name_property, prefix_block
            )
            if labels:
                shown = ", ".join(f'"{lbl}"' for lbl in labels[:12])
                more = "" if len(labels) <= 12 else f" (and {len(labels) - 12} more)"
                value_hints.append(f"{class_name} instances are named: {shown}{more}")

        if not path_hints and not value_hints:
            return ""

        sections = ["ABox discovery - actual data found in the endpoint:"]
        if path_hints:
            sections.append("Paths:")
            sections.extend(f"- {h}" for h in path_hints)
        if value_hints:
            sections.append("Actual entity names (use these exact strings in FILTER):")
            sections.extend(f"- {h}" for h in value_hints)
        sections.append(
            "If your FILTER text did not match any of the names above, "
            "adjust it to match one of the actual names."
        )
        return "\n".join(sections)
    except Exception:
        return ""


async def _discover_instance_labels(
    endpoint_url: str,
    class_uri: str,
    name_property: str,
    prefix_block: str,
) -> list[str]:
    """Fetch a sample of actual name/label values for instances of a class.

    Uses rdfs:label when the class uses rdfs:label, otherwise the custom name
    property from the naming strategy. Returns distinct string values.
    """
    if name_property == "rdfs:label":
        name_pattern = "?s rdfs:label ?name ."
    else:
        # Custom name property lives in the default namespace
        name_pattern = f"?s :{name_property} ?name ."

    sparql = f"""{prefix_block}
SELECT DISTINCT ?name WHERE {{
  ?s a <{class_uri}> .
  {name_pattern}
  FILTER(isLiteral(?name))
}}
LIMIT 25"""

    results = await _execute_discovery_query(endpoint_url, sparql)
    labels: list[str] = []
    seen = set()
    for row in results:
        value = row.get("name", {}).get("value", "")
        if value and value not in seen:
            seen.add(value)
            labels.append(value)
    return labels


async def _discover_outgoing_properties(
    endpoint_url: str,
    class_uri: str,
    prefix_block: str,
) -> list[dict[str, str]]:
    """Query for object properties actually used by instances of a class."""
    sparql = f"""{prefix_block}
SELECT DISTINCT ?p ?targetClass WHERE {{
  ?s a <{class_uri}> .
  ?s ?p ?o .
  ?o a ?targetClass .
  FILTER(?p != rdf:type)
  FILTER(?p != rdfs:label)
  FILTER(?p != rdfs:comment)
  FILTER(!STRSTARTS(STR(?targetClass), "http://www.w3.org/"))
}}
LIMIT 30"""

    results = await _execute_discovery_query(endpoint_url, sparql)
    props: list[dict[str, str]] = []
    seen = set()
    for row in results:
        prop = row.get("p", {}).get("value", "")
        target = row.get("targetClass", {}).get("value", "")
        if not prop or not target:
            continue
        key = f"{_local_name(prop)}->{_local_name(target)}"
        if key not in seen:
            seen.add(key)
            props.append({"property": _local_name(prop), "target_class": _local_name(target)})
    return props


async def _execute_discovery_query(endpoint_url: str, sparql: str) -> list[dict[str, object]]:
    try:
        async with httpx.AsyncClient(timeout=settings.fuseki_admin_timeout_seconds) as client:
            response = await client.post(
                endpoint_url,
                data={"query": sparql},
                headers={"Accept": "application/sparql-results+json, application/json"},
            )
            response.raise_for_status()
            return response.json().get("results", {}).get("bindings", [])
    except Exception:
        return []


def _extract_class_uris(query: str, ontology_context: dict[str, object]) -> list[str]:
    """Extract class URIs used in rdf:type / 'a' patterns in the query."""
    prefix_map = _build_prefix_map(ontology_context)
    known = _known_classes(ontology_context)
    found: list[str] = []

    patterns = [
        r"(?:rdf:type|(?<!\w)a)\s+:([A-Za-z_]\w*)",
        r"(?:rdf:type|(?<!\w)a)\s+([A-Za-z_][\w-]*):([A-Za-z_]\w*)",
        r"(?:rdf:type|(?<!\w)a)\s+<([^>]+)>",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, query):
            groups = match.groups()
            if len(groups) == 1:
                value = groups[0]
                if value.startswith("http"):
                    if value in known:
                        found.append(value)
                else:
                    default_ns = prefix_map.get("", "")
                    if default_ns and (default_ns + value) in known:
                        found.append(default_ns + value)
            elif len(groups) == 2:
                prefix, local = groups
                ns = prefix_map.get(prefix, "")
                if ns and (ns + local) in known:
                    found.append(ns + local)
    return list(dict.fromkeys(found))


def _build_prefix_map(ontology_context: dict[str, object]) -> dict[str, str]:
    prefixes = ontology_context.get("prefixes", [])
    if not isinstance(prefixes, list):
        return {}
    result: dict[str, str] = {}
    for item in prefixes:
        if not isinstance(item, dict):
            continue
        prefix = item.get("prefix")
        namespace = item.get("namespace")
        if isinstance(prefix, str) and isinstance(namespace, str):
            result["" if prefix == ":" else prefix] = namespace
    return result


def _build_prefix_block(ontology_context: dict[str, object]) -> str:
    lines = []
    for prefix, namespace in _build_prefix_map(ontology_context).items():
        if prefix == "":
            lines.append(f"PREFIX : <{namespace}>")
        else:
            lines.append(f"PREFIX {prefix}: <{namespace}>")
    return "\n".join(lines)


def _known_classes(ontology_context: dict[str, object]) -> set[str]:
    classes = ontology_context.get("classes", [])
    if not isinstance(classes, list):
        return set()
    return {
        str(item["uri"])
        for item in classes
        if isinstance(item, dict) and isinstance(item.get("uri"), str)
    }


def _local_name(uri: str) -> str:
    if "#" in uri:
        return uri.split("#")[-1]
    return uri.rstrip("/").rsplit("/", 1)[-1]
