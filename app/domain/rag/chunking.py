"""Chunk builders for ontology index construction."""

from __future__ import annotations


SUPPORTED_CHUNKING_STRATEGIES = {"class_based", "property_based", "composite"}
SUPPORTED_CHUNKING_ORDER = ("class_based", "property_based", "composite")
_DISPLAY_LIKE_GLOBAL_NAMES = {
    "altlabel",
    "alternative",
    "code",
    "description",
    "identifier",
    "label",
    "name",
    "preflabel",
    "title",
}
_INFRASTRUCTURE_NAMESPACES = {
    "http://purl.org/dc/elements/1.1/",
    "http://purl.org/dc/terms/",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "http://www.w3.org/2000/01/rdf-schema#",
    "http://www.w3.org/2002/07/owl#",
    "http://www.w3.org/2004/02/skos/core#",
}
_BUILTIN_VOCABULARY_NAMESPACES = {
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "http://www.w3.org/2000/01/rdf-schema#",
    "http://www.w3.org/2001/XMLSchema#",
    "http://www.w3.org/2002/07/owl#",
}


def _is_builtin_term(uri: object) -> bool:
    """True for RDF/RDFS/OWL/XSD vocabulary terms.

    These are language built-ins, not domain schema. Indexing them lets
    meaningless chunks such as owl:Restriction compete for the top-k retrieval
    slots, which measurably crowds out real classes on ontologies that declare
    the OWL vocabulary in their source file.
    """
    return isinstance(uri, str) and _namespace_for(uri) in _BUILTIN_VOCABULARY_NAMESPACES


def build_chunks(ontology_context: dict[str, object], chunking: str) -> list[dict[str, object]]:
    """Build retrieval chunks for the selected strategy."""
    if chunking == "class_based":
        return _build_class_based_chunks(ontology_context)
    if chunking == "property_based":
        return _build_property_based_chunks(ontology_context)
    if chunking == "composite":
        return _build_composite_chunks(ontology_context)
    raise ValueError(f"Unsupported chunking strategy: {chunking}")


def _build_class_based_chunks(ontology_context: dict[str, object]) -> list[dict[str, object]]:
    class_chunks: list[dict[str, object]] = []

    classes = ontology_context.get("classes", [])
    object_properties = ontology_context.get("object_properties", [])
    datatype_properties = ontology_context.get("datatype_properties", [])

    if not isinstance(classes, list):
        return class_chunks

    prefix_map = _prefix_map(ontology_context)
    class_names_by_uri = _class_names_by_uri(classes, prefix_map)
    superclass_closure = _superclass_closure_by_uri(classes)

    for class_data in classes:
        if not isinstance(class_data, dict):
            continue
        if _is_builtin_term(class_data.get("uri")):
            continue
        class_name = class_data.get("name") or class_data.get("label") or class_data.get("uri")
        if not class_name:
            continue

        class_uri = class_data.get("uri")
        class_uri_text = class_uri if isinstance(class_uri, str) else None
        class_name = _compact_uri(class_uri_text, prefix_map) if class_uri_text else _short_name(str(class_name))
        class_label = class_data.get("label")
        class_label = class_label.strip() if isinstance(class_label, str) else None
        description = _description_for(class_data, class_name)
        parent_classes = _named_uri_values(class_data.get("parent_classes"), class_names_by_uri, prefix_map)

        object_sections = _outgoing_property_sections_for_class(
            properties=object_properties if isinstance(object_properties, list) else [],
            class_uri=class_uri_text,
            superclass_uris=superclass_closure.get(class_uri_text or "", set()),
            class_names_by_uri=class_names_by_uri,
            prefix_map=prefix_map,
        )
        datatype_sections = _outgoing_property_sections_for_class(
            properties=datatype_properties if isinstance(datatype_properties, list) else [],
            class_uri=class_uri_text,
            superclass_uris=superclass_closure.get(class_uri_text or "", set()),
            class_names_by_uri=class_names_by_uri,
            prefix_map=prefix_map,
        )
        incoming_object_properties = _incoming_properties_for_class(
            properties=object_properties if isinstance(object_properties, list) else [],
            class_uri=class_uri_text,
            superclass_uris=superclass_closure.get(class_uri_text or "", set()),
            class_names_by_uri=class_names_by_uri,
            prefix_map=prefix_map,
        )

        text = _build_class_chunk_text(
            class_name=class_name,
            class_uri=class_uri_text,
            class_label=class_label,
            description=description,
            parent_classes=parent_classes,
            direct_object_properties=object_sections["direct"],
            inherited_object_properties=object_sections["inherited"],
            global_object_properties=object_sections["global"],
            incoming_object_properties=incoming_object_properties,
            direct_datatype_properties=datatype_sections["direct"],
            inherited_datatype_properties=datatype_sections["inherited"],
            global_datatype_properties=datatype_sections["global"],
        )

        class_chunks.append(
            {
                "chunk_type": "class",
                "class_name": class_name,
                "class_label": class_label,
                "class_uri": class_uri,
                "text": text,
                "metadata": {
                    "label": class_label,
                    "description": description,
                    "parent_classes": parent_classes,
                    "object_properties": object_sections["direct"],
                    "inherited_object_properties": object_sections["inherited"],
                    "global_object_properties": object_sections["global"],
                    "incoming_object_properties": incoming_object_properties,
                    "datatype_properties": datatype_sections["direct"],
                    "inherited_datatype_properties": datatype_sections["inherited"],
                    "global_datatype_properties": datatype_sections["global"],
                },
            }
        )

    return class_chunks


def _build_property_based_chunks(ontology_context: dict[str, object]) -> list[dict[str, object]]:
    property_chunks: list[dict[str, object]] = []
    classes = ontology_context.get("classes", [])
    prefix_map = _prefix_map(ontology_context)
    class_names_by_uri = _class_names_by_uri(classes if isinstance(classes, list) else [], prefix_map)
    properties = _all_properties(ontology_context)

    for property_data in properties:
        property_uri = property_data.get("uri")
        if _is_builtin_term(property_uri):
            continue
        property_name = property_data.get("name") or property_data.get("label") or property_uri
        if not property_name:
            continue

        property_name = _compact_uri(str(property_uri), prefix_map) if isinstance(property_uri, str) else _short_name(str(property_name))
        property_label = _clean_text(property_data.get("label"))
        description = _description_for(property_data, property_name)
        property_type = _clean_text(property_data.get("property_type")) or "property"
        domains = _named_uri_values(property_data.get("domain"), class_names_by_uri, prefix_map)
        ranges = _named_uri_values(property_data.get("range"), class_names_by_uri, prefix_map)
        inverse_properties = _named_uri_values(property_data.get("inverse_properties"), {}, prefix_map)
        usage_scope = _usage_scope_for_property(property_data)

        text = _build_property_chunk_text(
            property_name=property_name,
            property_uri=property_uri if isinstance(property_uri, str) else None,
            property_label=property_label,
            description=description,
            property_type=property_type,
            domains=domains,
            ranges=ranges,
            inverse_properties=inverse_properties,
            usage_scope=usage_scope,
        )

        property_chunks.append(
            {
                "chunk_type": "property",
                "property_name": property_name,
                "property_label": property_label,
                "property_uri": property_uri,
                "text": text,
                "metadata": {
                    "label": property_label,
                    "description": description,
                    "property_type": property_type,
                    "domain": domains,
                    "range": ranges,
                    "inverse_properties": inverse_properties,
                    "usage_scope": usage_scope,
                },
            }
        )

    return property_chunks


def _build_composite_chunks(ontology_context: dict[str, object]) -> list[dict[str, object]]:
    composite_chunks: list[dict[str, object]] = []
    classes = ontology_context.get("classes", [])
    object_properties = ontology_context.get("object_properties", [])
    datatype_properties = ontology_context.get("datatype_properties", [])

    if not isinstance(classes, list):
        return composite_chunks

    prefix_map = _prefix_map(ontology_context)
    children_by_uri = _child_classes_by_uri(classes)
    class_names_by_uri = _class_names_by_uri(classes, prefix_map)
    superclass_closure = _superclass_closure_by_uri(classes)

    for class_data in classes:
        if not isinstance(class_data, dict):
            continue
        if _is_builtin_term(class_data.get("uri")):
            continue
        class_uri = class_data.get("uri")
        class_name = class_data.get("name") or class_data.get("label") or class_uri
        if not class_name:
            continue

        class_uri_text = class_uri if isinstance(class_uri, str) else None
        class_name = _compact_uri(str(class_uri), prefix_map) if isinstance(class_uri, str) else _short_name(str(class_name))
        class_label = _clean_text(class_data.get("label"))
        description = _description_for(class_data, class_name)
        parent_classes = _named_uri_values(class_data.get("parent_classes"), class_names_by_uri, prefix_map)
        child_classes = [
            class_names_by_uri.get(child_uri, _compact_uri(child_uri, prefix_map))
            for child_uri in children_by_uri.get(class_uri_text or "", [])
        ]
        object_sections = _outgoing_property_sections_for_class(
            properties=object_properties if isinstance(object_properties, list) else [],
            class_uri=class_uri_text,
            superclass_uris=superclass_closure.get(class_uri_text or "", set()),
            class_names_by_uri=class_names_by_uri,
            prefix_map=prefix_map,
        )
        datatype_sections = _outgoing_property_sections_for_class(
            properties=datatype_properties if isinstance(datatype_properties, list) else [],
            class_uri=class_uri_text,
            superclass_uris=superclass_closure.get(class_uri_text or "", set()),
            class_names_by_uri=class_names_by_uri,
            prefix_map=prefix_map,
        )
        incoming_object_properties = _incoming_properties_for_class(
            properties=object_properties if isinstance(object_properties, list) else [],
            class_uri=class_uri_text,
            superclass_uris=superclass_closure.get(class_uri_text or "", set()),
            class_names_by_uri=class_names_by_uri,
            prefix_map=prefix_map,
        )

        text = _build_composite_chunk_text(
            class_name=class_name,
            class_uri=class_uri_text,
            class_label=class_label,
            description=description,
            parent_classes=parent_classes,
            child_classes=child_classes,
            direct_object_properties=object_sections["direct"],
            inherited_object_properties=object_sections["inherited"],
            global_object_properties=object_sections["global"],
            direct_datatype_properties=datatype_sections["direct"],
            inherited_datatype_properties=datatype_sections["inherited"],
            global_datatype_properties=datatype_sections["global"],
            incoming_object_properties=incoming_object_properties,
        )

        composite_chunks.append(
            {
                "chunk_type": "composite",
                "class_name": class_name,
                "class_label": class_label,
                "class_uri": class_uri,
                "text": text,
                "metadata": {
                    "label": class_label,
                    "description": description,
                    "parent_classes": parent_classes,
                    "child_classes": child_classes,
                    "object_properties": object_sections["direct"],
                    "inherited_object_properties": object_sections["inherited"],
                    "global_object_properties": object_sections["global"],
                    "datatype_properties": datatype_sections["direct"],
                    "inherited_datatype_properties": datatype_sections["inherited"],
                    "global_datatype_properties": datatype_sections["global"],
                    "incoming_object_properties": incoming_object_properties,
                },
            }
        )

    return composite_chunks


def _all_properties(ontology_context: dict[str, object]) -> list[dict[str, object]]:
    properties: list[dict[str, object]] = []
    for key in ("object_properties", "datatype_properties"):
        value = ontology_context.get(key, [])
        if not isinstance(value, list):
            continue
        properties.extend(item for item in value if isinstance(item, dict))
    return properties


def _class_names_by_uri(classes: list[dict[str, object]], prefix_map: dict[str, str] | None = None) -> dict[str, str]:
    names: dict[str, str] = {}
    for class_data in classes:
        class_uri = class_data.get("uri")
        class_name = class_data.get("name") or class_data.get("label") or class_uri
        if isinstance(class_uri, str) and class_uri and class_name:
            names[class_uri] = _compact_uri(class_uri, prefix_map or {})
    return names


def _child_classes_by_uri(classes: list[dict[str, object]]) -> dict[str, list[str]]:
    child_map: dict[str, list[str]] = {}
    for class_data in classes:
        child_uri = class_data.get("uri")
        if not isinstance(child_uri, str) or not child_uri:
            continue
        for parent_uri in class_data.get("parent_classes", []):
            if isinstance(parent_uri, str) and parent_uri:
                child_map.setdefault(parent_uri, []).append(child_uri)
    return {parent_uri: sorted(child_uris) for parent_uri, child_uris in child_map.items()}


def _superclass_closure_by_uri(classes: list[dict[str, object]]) -> dict[str, set[str]]:
    direct_parents: dict[str, set[str]] = {}
    for class_data in classes:
        class_uri = class_data.get("uri")
        if not isinstance(class_uri, str) or not class_uri:
            continue
        direct_parents[class_uri] = {
            parent_uri
            for parent_uri in class_data.get("parent_classes", [])
            if isinstance(parent_uri, str) and parent_uri
        }

    closure: dict[str, set[str]] = {}

    def collect(class_uri: str, seen: set[str]) -> set[str]:
        if class_uri in closure:
            return closure[class_uri]
        parents = direct_parents.get(class_uri, set())
        result: set[str] = set()
        for parent_uri in parents:
            if parent_uri in seen:
                continue
            result.add(parent_uri)
            result.update(collect(parent_uri, seen | {parent_uri}))
        closure[class_uri] = result
        return result

    for class_uri in direct_parents:
        collect(class_uri, {class_uri})
    return closure


def _outgoing_property_sections_for_class(
    properties: list[dict[str, object]],
    class_uri: str | None,
    superclass_uris: set[str],
    class_names_by_uri: dict[str, str],
    prefix_map: dict[str, str],
) -> dict[str, list[str]]:
    sections = {"direct": [], "inherited": [], "global": []}
    if not class_uri:
        return sections

    for property_data in properties:
        domains = property_data.get("domain", [])
        if not isinstance(domains, list):
            continue

        if class_uri in domains:
            sections["direct"].append(_property_line(property_data, "range", class_names_by_uri, prefix_map))
            continue

        inherited_from = sorted(superclass_uris.intersection(item for item in domains if isinstance(item, str)))
        if inherited_from:
            source_text = ", ".join(class_names_by_uri.get(uri, _compact_uri(uri, prefix_map)) for uri in inherited_from)
            sections["inherited"].append(
                f"{_property_line(property_data, 'range', class_names_by_uri, prefix_map)} (from {source_text})"
            )
            continue

        if not domains and _should_propagate_global_property(property_data, class_names_by_uri):
            sections["global"].append(_property_line(property_data, "range", class_names_by_uri, prefix_map))

    return {key: sorted(set(values)) for key, values in sections.items()}


def _incoming_properties_for_class(
    properties: list[dict[str, object]],
    class_uri: str | None,
    superclass_uris: set[str],
    class_names_by_uri: dict[str, str],
    prefix_map: dict[str, str],
) -> list[str]:
    """Find properties where this class appears as the range (i.e., is the target).

    Returns lines like: "providesTrainingCourse (from TrainingCentre)"
    """
    if not class_uri:
        return []

    incoming = []
    for property_data in properties:
        ranges = property_data.get("range", [])
        if not isinstance(ranges, list):
            continue
        is_direct = class_uri in ranges
        inherited_from = sorted(superclass_uris.intersection(item for item in ranges if isinstance(item, str)))
        if not is_direct and not inherited_from:
            continue
        property_name = property_data.get("name") or property_data.get("label") or property_data.get("uri")
        if not property_name:
            continue
        domains = property_data.get("domain", [])
        source_names = []
        if isinstance(domains, list):
            for domain_uri in domains:
                if isinstance(domain_uri, str):
                    source_names.append(class_names_by_uri.get(domain_uri, _compact_uri(domain_uri, prefix_map)))
        source_text = ", ".join(source_names) if source_names else "unknown source"
        property_uri = property_data.get("uri")
        property_text = _compact_uri(property_uri, prefix_map) if isinstance(property_uri, str) else _short_name(str(property_name))
        line = f"{property_text} <- {source_text}"
        inverse_text = _inverse_text(property_data, prefix_map)
        if inverse_text:
            line = f"{line} (inverse: {inverse_text})"
        if inherited_from:
            inherited_text = ", ".join(class_names_by_uri.get(uri, _compact_uri(uri, prefix_map)) for uri in inherited_from)
            line = f"{line} (range inherited from {inherited_text})"
        incoming.append(line)
    return sorted(set(incoming))


def _property_line(
    property_data: dict[str, object],
    target_key: str,
    class_names_by_uri: dict[str, str],
    prefix_map: dict[str, str],
) -> str:
    property_name = property_data.get("name") or property_data.get("label") or property_data.get("uri")
    target_values = _named_uri_values(property_data.get(target_key), class_names_by_uri, prefix_map)
    target_text = ", ".join(target_values) if target_values else "Unknown"
    property_uri = property_data.get("uri")
    property_text = _compact_uri(property_uri, prefix_map) if isinstance(property_uri, str) else _short_name(str(property_name))
    line = f"{property_text} -> {target_text}"
    inverse_text = _inverse_text(property_data, prefix_map)
    if inverse_text:
        line = f"{line} (inverse: {inverse_text})"
    return line


def _inverse_text(property_data: dict[str, object], prefix_map: dict[str, str]) -> str:
    inverse_values = property_data.get("inverse_properties", [])
    if not isinstance(inverse_values, list):
        return ""
    names = sorted({_compact_uri(str(value), prefix_map) for value in inverse_values if isinstance(value, str) and value})
    return ", ".join(names)


def _usage_scope_for_property(property_data: dict[str, object]) -> str:
    domains = property_data.get("domain", [])
    ranges = property_data.get("range", [])
    has_domain = isinstance(domains, list) and bool(domains)
    has_range = isinstance(ranges, list) and bool(ranges)
    if has_domain and has_range:
        return "direct domain/range from ontology"
    if has_domain:
        return "direct domain from ontology; range unspecified"
    if has_range:
        return "no explicit domain; globally available candidate property with declared range"
    return "no explicit domain or range; globally available candidate property"


def _should_propagate_global_property(
    property_data: dict[str, object],
    class_names_by_uri: dict[str, str],
) -> bool:
    property_uri = property_data.get("uri")
    property_name = property_data.get("name") or property_data.get("label") or property_uri
    if not isinstance(property_uri, str) or not property_uri:
        return True

    property_namespace = _namespace_for(property_uri)
    if property_namespace in _INFRASTRUCTURE_NAMESPACES:
        return bool(property_name and _short_name(str(property_name)).lower() in _DISPLAY_LIKE_GLOBAL_NAMES)

    class_namespaces = {_namespace_for(class_uri) for class_uri in class_names_by_uri}
    if property_namespace in class_namespaces:
        return True

    if property_name and _short_name(str(property_name)).lower() in _DISPLAY_LIKE_GLOBAL_NAMES:
        return True

    return False


def _build_class_chunk_text(
    *,
    class_name: str,
    class_uri: str | None,
    class_label: str | None,
    description: str,
    parent_classes: list[str],
    direct_object_properties: list[str],
    inherited_object_properties: list[str],
    global_object_properties: list[str],
    incoming_object_properties: list[str],
    direct_datatype_properties: list[str],
    inherited_datatype_properties: list[str],
    global_datatype_properties: list[str],
) -> str:
    description_text = description or "No description available."
    label_text = class_label or "No label available."
    uri_text = class_uri or "Unknown"

    return (
        f"Class: {class_name}\n\n"
        f"Class URI: {uri_text}\n\n"
        f"Label: {label_text}\n\n"
        f"Description: {description_text}\n\n"
        f"Parent Classes:\n{_bullet_list(parent_classes, empty_label='None (top-level class)')}\n\n"
        f"Direct Object Properties:\n{_bullet_list(direct_object_properties)}\n\n"
        f"Inherited Object Properties:\n{_bullet_list(inherited_object_properties)}\n\n"
        f"Global Object Properties:\n{_bullet_list(global_object_properties)}\n\n"
        f"Incoming Object Properties (this class is the target):\n{_bullet_list(incoming_object_properties)}\n\n"
        f"Direct Datatype Properties:\n{_bullet_list(direct_datatype_properties)}\n\n"
        f"Inherited Datatype Properties:\n{_bullet_list(inherited_datatype_properties)}\n\n"
        f"Global Datatype Properties:\n{_bullet_list(global_datatype_properties)}"
    )


def _build_property_chunk_text(
    *,
    property_name: str,
    property_uri: str | None,
    property_label: str | None,
    description: str,
    property_type: str,
    domains: list[str],
    ranges: list[str],
    inverse_properties: list[str],
    usage_scope: str,
) -> str:
    label_text = property_label or "No label available."
    description_text = description or "No description available."
    uri_text = property_uri or "Unknown"
    return (
        f"Property: {property_name}\n\n"
        f"Property URI: {uri_text}\n\n"
        f"Label: {label_text}\n\n"
        f"Description: {description_text}\n\n"
        f"Type: {property_type}\n\n"
        f"Domain Classes:\n{_bullet_list(domains)}\n\n"
        f"Range Classes or Datatypes:\n{_bullet_list(ranges)}\n\n"
        f"Inverse Properties:\n{_bullet_list(inverse_properties)}\n\n"
        f"Usage Scope:\n- {usage_scope}"
    )


def _build_composite_chunk_text(
    *,
    class_name: str,
    class_uri: str | None,
    class_label: str | None,
    description: str,
    parent_classes: list[str],
    child_classes: list[str],
    direct_object_properties: list[str],
    inherited_object_properties: list[str],
    global_object_properties: list[str],
    direct_datatype_properties: list[str],
    inherited_datatype_properties: list[str],
    global_datatype_properties: list[str],
    incoming_object_properties: list[str] | None = None,
) -> str:
    label_text = class_label or "No label available."
    description_text = description or "No description available."
    incoming_text = _bullet_list(incoming_object_properties or [])
    uri_text = class_uri or "Unknown"
    return (
        f"Class Neighbourhood: {class_name}\n\n"
        f"Class URI: {uri_text}\n\n"
        f"Label: {label_text}\n\n"
        f"Description: {description_text}\n\n"
        f"Parent Classes:\n{_bullet_list(parent_classes, empty_label='None (top-level class)')}\n\n"
        f"Child Classes:\n{_bullet_list(child_classes)}\n\n"
        f"Direct Object Properties:\n{_bullet_list(direct_object_properties)}\n\n"
        f"Inherited Object Properties:\n{_bullet_list(inherited_object_properties)}\n\n"
        f"Global Object Properties:\n{_bullet_list(global_object_properties)}\n\n"
        f"Incoming Object Properties (this class is the target):\n{incoming_text}\n\n"
        f"Direct Datatype Properties:\n{_bullet_list(direct_datatype_properties)}\n\n"
        f"Inherited Datatype Properties:\n{_bullet_list(inherited_datatype_properties)}\n\n"
        f"Global Datatype Properties:\n{_bullet_list(global_datatype_properties)}"
    )


def _description_for(class_data: dict[str, object], class_name: str) -> str:
    comment = class_data.get("comment")
    description = comment.strip() if isinstance(comment, str) else ""
    if description:
        return description

    label = class_data.get("label")
    if isinstance(label, str):
        normalized = label.strip()
        if normalized and normalized != class_name:
            return normalized

    return "No description available"


def _named_uri_values(value: object, class_names_by_uri: dict[str, str], prefix_map: dict[str, str]) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted(
        {
            class_names_by_uri.get(item, _compact_uri(item, prefix_map))
            for item in value
            if isinstance(item, str) and item
        }
    )


def _prefix_map(ontology_context: dict[str, object]) -> dict[str, str]:
    prefixes = ontology_context.get("prefixes", [])
    if not isinstance(prefixes, list):
        return {}
    result: dict[str, str] = {}
    for item in prefixes:
        if not isinstance(item, dict):
            continue
        prefix = item.get("prefix")
        namespace = item.get("namespace")
        if isinstance(prefix, str) and isinstance(namespace, str) and namespace:
            result["" if prefix == ":" else prefix] = namespace
    return result


def _compact_uri(uri: str, prefix_map: dict[str, str]) -> str:
    if not uri:
        return ""
    best_prefix = None
    best_namespace = ""
    for prefix, namespace in prefix_map.items():
        if uri.startswith(namespace) and len(namespace) > len(best_namespace):
            best_prefix = prefix
            best_namespace = namespace
    if best_prefix is not None:
        local = uri[len(best_namespace):]
        return f":{local}" if best_prefix == "" else f"{best_prefix}:{local}"
    if "://" in uri:
        return f"<{uri}>"
    return _short_name(uri)


def _clean_text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _short_name(value: str) -> str:
    if "#" in value:
        return value.rsplit("#", 1)[-1]
    if "/" in value:
        return value.rstrip("/").rsplit("/", 1)[-1]
    if ":" in value:
        return value.rsplit(":", 1)[-1]
    return value


def _namespace_for(value: str) -> str:
    if "#" in value:
        return value.rsplit("#", 1)[0] + "#"
    if "/" in value:
        return value.rstrip("/").rsplit("/", 1)[0] + "/"
    return ""


def _bullet_list(values: list[str], empty_label: str = "None") -> str:
    if not values:
        return f"- {empty_label}"
    return "\n".join(f"- {value}" for value in values)