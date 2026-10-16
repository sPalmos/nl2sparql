"""Render runtime LLM prompts from package and retrieval context.

This module is the only runtime layer that knows about the Jinja templates. It
turns package metadata, retrieved RAG chunks, ontology prefixes, failed queries,
and validation/execution feedback into prompt strings. It does not call the LLM
or decide when prompts should be rendered.

Prompt rules are ontology-adaptive: the module detects whether the ontology uses
rdfs:label or custom name properties and selects the appropriate entity matching
and result shape rules automatically.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.domain.rag import RetrievedABoxChunk, RetrievedChunk


SYSTEM_ROLE = (
    "You are an expert SPARQL query generator. "
    "Use only the provided ontology context and URIs. "
    "Do not invent classes, properties, or namespaces."
)

CORRECTION_SYSTEM_ROLE = (
    "You are an expert SPARQL query generator. "
    "Your previous query failed validation. "
    "Read the error messages carefully - they identify the exact problem. "
    "Read the ontology chunks carefully - they contain the correct class and property names. "
    "Fix the query using only URIs that exist in the provided ontology context."
)

PREFIX_USAGE_RULES = """Prefix Usage Rules:
- Use only the prefix declarations listed above.
- Do not use the ontology label or dataset label as a prefix.
- Use the exact prefixed class and property terms shown in the retrieved ontology chunks.
- Class and property terms may come from different namespaces; do not assume a property uses `:` because its class uses `:`.
- If a default prefix declaration is listed as `PREFIX : <...>`, use `:` only for terms that belong to that exact namespace.
- Unknown prefixes will fail validation.
"""

# ── Adaptive Entity Matching Rules ────────────────────────────────────────────

_ENTITY_MATCHING_RULES_RDFS_LABEL = """
Entity Matching Rules:
- Never construct individual/instance URIs directly from resource names.
- Instance URIs are opaque identifiers that cannot be guessed from labels.
- If Matching Instance Candidates contain an unambiguous URI for a resource explicitly mentioned in the question, you may use that exact URI as a fixed resource in the query.
- Otherwise, find instances by class and label using this pattern:
  ?entity rdf:type :ClassName ; rdfs:label ?label .
  FILTER(CONTAINS(LCASE(STR(?label)), "search term"))
- If a property links to another resource/class, do not compare that resource variable to string literals. Bind the resource and compare its label/name value instead.
- Use LCASE and CONTAINS for partial, case-insensitive matching.
- When the question names a specific entity, extract the key words for the FILTER.
- When no specific entity is named, omit the FILTER to match all instances."""

_ENTITY_MATCHING_RULES_CUSTOM = """
Entity Matching Rules:
- Never construct individual/instance URIs directly.
- Instance URIs are opaque identifiers that cannot be guessed.
- This ontology does NOT use rdfs:label. Do not use rdfs:label for entity names.
- If Matching Instance Candidates contain an unambiguous URI for a resource explicitly mentioned in the question, you may use that exact URI as a fixed resource in the query.
- Instead, use the domain-specific name properties from the ontology chunks.
- Name properties detected in this ontology:
{name_property_hints}
- To find a specific entity by name, use the appropriate name property with FILTER:
  ?entity rdf:type [exact class term from chunks] ;
          [exact name property term from chunks] ?name .
  FILTER(CONTAINS(LCASE(STR(?name)), "search term"))
- If a property links to another resource/class, do not compare that resource variable to string literals. Bind the resource and compare its name value instead.
- Use LCASE and CONTAINS for partial, case-insensitive matching.
- When no specific entity is named, omit the FILTER to match all instances."""

_ENTITY_MATCHING_RULES_MIXED = """
Entity Matching Rules:
- Never construct individual/instance URIs directly from resource names.
- Instance URIs are opaque identifiers that cannot be guessed from labels.
- If Matching Instance Candidates contain an unambiguous URI for a resource explicitly mentioned in the question, you may use that exact URI as a fixed resource in the query.
- Most classes in this ontology use rdfs:label. As the default fallback, find instances by class and label:
  ?entity rdf:type [exact class term from chunks] ; rdfs:label ?label .
  FILTER(CONTAINS(LCASE(STR(?label)), "search term"))
- EXCEPTION — the following classes do NOT use rdfs:label. For these, use the listed name property instead of rdfs:label:
{name_property_exceptions}
- If a property links to another resource/class, do not compare that resource variable to string literals. Bind the resource and compare its label/name value instead.
- Use LCASE and CONTAINS for partial, case-insensitive matching.
- When the question names a specific entity, extract the key words for the FILTER.
- When no specific entity is named, omit the FILTER to match all instances."""

# ── Adaptive Result Shape Rules ───────────────────────────────────────────────

_RESULT_SHAPE_RULES_RDFS_LABEL = """Result Shape Rules:
- If the answer includes an ontology entity/resource and `rdfs:label` is available, return the label variable instead of the entity URI.
- Use `rdfs:label` for labels when the `rdfs:` prefix is listed above.
- Use `skos:prefLabel` as another label option only when the `skos:` prefix is listed above.
- For aggregate queries grouped by an ontology entity/resource, do not return only the grouping URI. Join the resource to its label and return the label variable as the displayed answer. Group by both the resource and the label when needed.
- Use URI variables as internal join/grouping helpers when needed, but do not include those helper URI variables in SELECT unless the question explicitly asks for URIs.
- If a label might not exist, use `OPTIONAL` and `COALESCE` to expose one display variable rather than selecting both the entity URI and label.
- Return an entity URI only when the question explicitly asks for URIs or no label predicate is available.
- Do not invent label properties or label prefixes."""

_RESULT_SHAPE_RULES_CUSTOM = """Result Shape Rules:
- This ontology does NOT use rdfs:label for display names. Do not use rdfs:label.
- When returning entity names, use the exact domain-specific name property term from the ontology chunks.
- For aggregate queries grouped by an entity, expose the name property and group by both the resource and the name value.
- Use URI variables as internal join/grouping helpers when needed, but do not include those helper URI variables in SELECT unless the question explicitly asks for URIs.
- If no name property is visible for a class, return the entity URI.
- Do not invent name or label properties."""

_RESULT_SHAPE_EXAMPLE_RDFS_LABEL = """Result Shape Example:
For questions that count or group resources, expose the resource label:
...
?entity rdfs:label ?entityLabel .
...
GROUP BY ?entityLabel
"""

_RESULT_SHAPE_EXAMPLE_CUSTOM = """Result Shape Example:
For questions that count or group resources, expose the resource name:
...
?entity :nameProperty ?entityName .
...
GROUP BY ?entityName
"""

STRICT_CONSTRAINTS = """Strict Constraints:
- Only use class and property terms that appear in the Relevant Ontology Chunks above.
- If the exact prefixed property term is not visible in the chunks, re-read them carefully before writing the query. Do not guess or invent property terms.
- Every variable in SELECT must appear in the WHERE clause.
- SELECT only the answer variables requested by the question. Do not include helper variables used only for joins, grouping, filtering, or URI grounding.
- Do not use OPTIONAL unless the question explicitly implies some data may be missing.
"""

ABOX_GROUNDING_RULES = """ABox Grounding Rules:
- Matching Instance Candidates are examples from the data, not the full answer set.
- Use a candidate URI only when it clearly matches a resource explicitly named in the question.
- Do not use candidate URIs to limit the variable being asked for in broad list/count/comparison questions.
- If no candidate clearly matches a named resource, use general graph patterns and the ontology name/label rules."""

OUTPUT_FORMAT_INSTRUCTIONS = """Output Format Instructions:
- Return only one valid SPARQL query.
- Use either full URIs in angle brackets or the provided prefix declarations.
- Only prefixes listed under Auto-Generated Prefix Declarations are allowed.
- Ontology and dataset names are labels, not SPARQL prefixes.
- Use the ':' prefix for terms in the default ontology namespace when it is listed.
- Do not invent prefixes, classes, properties, or namespaces.
- Do not include explanations, markdown fences, or extra text."""

CORRECTION_OUTPUT_FORMAT_INSTRUCTIONS = (
    "Return only a corrected SPARQL query. "
    "Use either full URIs in angle brackets or the available prefix declarations. "
    "Do not invent prefixes, classes, properties, or namespaces. "
    "Do not include explanations, markdown fences, or extra text."
)


# ── Adaptive Rule Builder ─────────────────────────────────────────────────────


def build_prompt_rules(
    ontology_context: dict[str, object],
    *,
    use_adaptive_naming: bool = True,
    use_full_rules: bool = True,
) -> str:
    """Build prompt rules adapted to the ontology's labeling strategy.

    Reads the data-driven `naming_strategy` computed during onboarding (which
    inspects actual instance data per class). Three cases:

    - "rdfs:label": all classes use rdfs:label → rdfs:label rules
    - "custom": no class uses rdfs:label → custom-name rules with per-class hints
    - "mixed": most classes use rdfs:label, a few use custom names → rdfs:label
      rules as the primary instruction, PLUS a note listing the exception classes

    Falls back to a schema heuristic only for older packages without
    naming_strategy.

    `use_adaptive_naming=False` ignores the detected strategy and always emits
    the rdfs:label rules, which is the assumption a non-adaptive system makes.
    `use_full_rules=False` emits only prefix usage and strict constraints,
    dropping the entity-matching, result-shape and ABox grounding blocks.
    """
    if not use_adaptive_naming:
        return _render_rules(mode="rdfs_label", name_props=[], full_rules=use_full_rules)

    naming = ontology_context.get("naming_strategy")

    if isinstance(naming, dict) and "default" in naming:
        default = naming.get("default")
        per_class = naming.get("per_class", {})
        custom_props = _name_props_from_strategy(per_class, ontology_context)

        if default == "custom":
            return _render_rules(mode="custom", name_props=custom_props, full_rules=use_full_rules)
        if default == "mixed":
            # rdfs:label is primary; list the custom-name exceptions
            return _render_rules(mode="mixed", name_props=custom_props, full_rules=use_full_rules)
        # default == "rdfs:label"
        return _render_rules(mode="rdfs_label", name_props=[], full_rules=use_full_rules)

    # Fallback for packages onboarded before naming_strategy existed
    name_props = _find_name_properties(ontology_context)
    use_custom = _should_use_custom_naming(name_props, ontology_context)
    return _render_rules(
        mode="custom" if use_custom else "rdfs_label",
        name_props=name_props if use_custom else [],
        full_rules=use_full_rules,
    )


def _render_rules(
    *,
    mode: str,
    name_props: list[dict[str, str]],
    full_rules: bool = True,
) -> str:
    """Assemble the prompt rules for the given naming mode.

    With `full_rules=False` only the prefix-usage and strict-constraint blocks
    are emitted; entity-matching, result-shape and ABox grounding guidance are
    omitted so their joint contribution can be measured.
    """
    if not full_rules:
        sections = (PREFIX_USAGE_RULES, STRICT_CONSTRAINTS)
        return "\n\n".join(section.strip() for section in sections) + "\n"

    if mode == "custom":
        hints = _build_name_property_hints(name_props)
        entity_rules = _ENTITY_MATCHING_RULES_CUSTOM.replace("{name_property_hints}", hints)
        result_rules = _RESULT_SHAPE_RULES_CUSTOM
        result_example = _RESULT_SHAPE_EXAMPLE_CUSTOM
    elif mode == "mixed":
        exceptions = _build_name_property_hints(name_props)
        entity_rules = _ENTITY_MATCHING_RULES_MIXED.replace("{name_property_exceptions}", exceptions)
        result_rules = _RESULT_SHAPE_RULES_RDFS_LABEL
        result_example = _RESULT_SHAPE_EXAMPLE_RDFS_LABEL
    else:  # rdfs_label
        entity_rules = _ENTITY_MATCHING_RULES_RDFS_LABEL
        result_rules = _RESULT_SHAPE_RULES_RDFS_LABEL
        result_example = _RESULT_SHAPE_EXAMPLE_RDFS_LABEL

    sections = (
        PREFIX_USAGE_RULES,
        entity_rules,
        result_rules,
        result_example,
        ABOX_GROUNDING_RULES,
        STRICT_CONSTRAINTS,
    )
    return "\n\n".join(section.strip() for section in sections) + "\n"


def _name_props_from_strategy(per_class: dict, ontology_context: dict[str, object]) -> list[dict[str, str]]:
    """Convert the stored per-class naming map into name-property hint entries.

    Only includes classes that use a custom name property (not rdfs:label),
    so the hints tell the LLM exactly which property to use for each class.
    """
    result: list[dict[str, str]] = []
    if not isinstance(per_class, dict):
        return result
    for class_name, name_property in per_class.items():
        if isinstance(name_property, str) and name_property != "rdfs:label":
            result.append(
                {
                    "class": _compact_class_term(str(class_name), ontology_context),
                    "property": _compact_property_term(str(name_property), str(class_name), ontology_context),
                }
            )
    return result


def _should_use_custom_naming(
    name_props: list[dict[str, str]],
    ontology_context: dict[str, object],
    coverage_threshold: float = 0.25,
) -> bool:
    """Decide whether custom name properties are the primary labeling mechanism.

    If name-like properties cover at least 25% of the ontology's classes,
    the ontology uses custom naming. Otherwise, rdfs:label is assumed.

    Examples:
    - eNOVATION: 110 classes, name props cover 1 (Person) → 0.9% → rdfs:label
    - Dog Kennels: 8 classes, name props cover 4 → 50% → custom
    """
    if not name_props:
        return False

    total_classes = len(ontology_context.get("classes", []))
    if total_classes == 0:
        return False

    # Count unique classes covered by name properties
    covered_classes = {p["class"] for p in name_props if p["class"] != "unknown"}
    coverage = len(covered_classes) / total_classes

    return coverage >= coverage_threshold


def _find_name_properties(
    ontology_context: dict[str, object],
) -> list[dict[str, str]]:
    """Find datatype properties that look like name/label/title fields.

    Returns a list of {"property": compactTerm, "class": compactTerm} dicts.
    These are injected into the custom entity matching rules so the LLM
    knows exactly which property to use for each class.
    """
    name_keywords = {"name", "label", "title"}
    # Exclude properties that are IDs, codes, or other non-name fields
    exclude_keywords = {"code", "id", "type", "date", "number", "phone", "email",
                        "address", "street", "city", "state", "zip", "amount",
                        "cost", "weight", "age", "gender", "description", "yn"}
    results: list[dict[str, str]] = []
    seen = set()

    for prop in ontology_context.get("datatype_properties", []):
        if not isinstance(prop, dict):
            continue
        prop_name = (prop.get("name") or "").lower()
        prop_uri = prop.get("uri", "")

        # Check if any name keyword is in the property name
        has_name_keyword = any(kw in prop_name for kw in name_keywords)
        # Check it's not actually an ID/code field that happens to contain "name"
        is_excluded = any(kw in prop_name for kw in exclude_keywords)

        if has_name_keyword and not is_excluded:
            domains = prop.get("domain", [])
            class_name = "unknown"
            if isinstance(domains, list) and domains:
                first_domain = domains[0]
                if isinstance(first_domain, str):
                    class_name = _compact_uri(first_domain, _prefix_map(ontology_context))

            local = _compact_uri(prop_uri, _prefix_map(ontology_context)) if prop_uri else prop.get("name", "")
            key = f"{class_name}.{local}"
            if key not in seen:
                seen.add(key)
                results.append({"property": local, "class": class_name})

    return results


def _build_name_property_hints(name_props: list[dict[str, str]]) -> str:
    """Format detected name properties as concrete hints for the prompt."""
    lines = []
    for prop in name_props:
        lines.append(f"  - {prop['class']}: use {prop['property']}")
    return "\n".join(lines) if lines else "  - (check ontology chunks for name properties)"


def _compact_class_term(class_name: str, ontology_context: dict[str, object]) -> str:
    prefix_map = _prefix_map(ontology_context)
    for class_data in ontology_context.get("classes", []):
        if not isinstance(class_data, dict):
            continue
        uri = class_data.get("uri")
        local = class_data.get("name")
        if isinstance(uri, str) and (class_name == local or class_name == _local_name(uri) or class_name == uri):
            return _compact_uri(uri, prefix_map)
    return class_name


def _compact_property_term(property_name: str, class_name: str, ontology_context: dict[str, object]) -> str:
    prefix_map = _prefix_map(ontology_context)
    class_uri = _class_uri_for_name(class_name, ontology_context)
    fallback = property_name
    for prop in ontology_context.get("datatype_properties", []):
        if not isinstance(prop, dict):
            continue
        uri = prop.get("uri")
        local = prop.get("name")
        if not isinstance(uri, str):
            continue
        if property_name not in {uri, local, _local_name(uri)}:
            continue
        compact = _compact_uri(uri, prefix_map)
        fallback = compact
        domains = prop.get("domain", [])
        if class_uri and isinstance(domains, list) and class_uri in domains:
            return compact
    return fallback


def _class_uri_for_name(class_name: str, ontology_context: dict[str, object]) -> str | None:
    for class_data in ontology_context.get("classes", []):
        if not isinstance(class_data, dict):
            continue
        uri = class_data.get("uri")
        local = class_data.get("name")
        if isinstance(uri, str) and (class_name == uri or class_name == local or class_name == _local_name(uri)):
            return uri
    return None


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
    return uri


def _local_name(uri: str) -> str:
    """Extract local name from a URI."""
    if "#" in uri:
        return uri.split("#")[-1]
    if "/" in uri:
        return uri.rstrip("/").rsplit("/", 1)[-1]
    return uri


# ── Render Functions ──────────────────────────────────────────────────────────


def render_query_generation_prompt(
    *,
    question: str,
    retrieved_context: list[RetrievedChunk],
    retrieved_abox_context: list[RetrievedABoxChunk] | None = None,
    metadata: dict[str, object],
    ontology_context: dict[str, object],
    few_shot_examples: list[dict[str, str]] | None = None,
    use_adaptive_naming: bool = True,
    use_full_rules: bool = True,
) -> str:
    """Render the first-query generation prompt.

    Prompt rules are automatically adapted based on the ontology's labeling
    strategy (rdfs:label vs custom name properties).
    """
    template = _template_environment().get_template("query_generation_prompt.j2")
    return template.render(
        system_role=SYSTEM_ROLE,
        ontology_name=metadata.get("ontology_name") if isinstance(metadata.get("ontology_name"), str) else None,
        dataset_name=metadata.get("dataset_name") if isinstance(metadata.get("dataset_name"), str) else None,
        retrieved_context=[{"rank": item.rank, "text": item.text} for item in retrieved_context],
        retrieved_abox_context=[
            {"rank": item.rank, "uri": item.uri, "display_name": item.display_name, "types": item.types, "text": item.text}
            for item in (retrieved_abox_context or [])
        ],
        prefix_declarations=prefix_declarations(ontology_context),
        prompt_rules=build_prompt_rules(
            ontology_context,
            use_adaptive_naming=use_adaptive_naming,
            use_full_rules=use_full_rules,
        ),
        few_shot_examples=few_shot_examples or [],
        output_format_instructions=OUTPUT_FORMAT_INSTRUCTIONS,
        user_question=question.strip(),
    )


def render_correction_prompt(
    *,
    original_question: str,
    failed_query: str,
    validation_errors: list[str],
    retrieved_context: list[RetrievedChunk] | list[dict[str, object]] | None = None,
    retrieved_abox_context: list[RetrievedABoxChunk] | list[dict[str, object]] | None = None,
    ontology_context: dict[str, object] | None = None,
    use_adaptive_naming: bool = True,
    use_full_rules: bool = True,
) -> str:
    """Render the correction prompt for one failed runtime attempt."""
    template = _template_environment().get_template("query_correction_prompt.j2")
    return template.render(
        system_role=CORRECTION_SYSTEM_ROLE,
        original_question=original_question.strip(),
        failed_query=failed_query.strip(),
        validation_errors=validation_errors,
        retrieved_context=_retrieved_context_payload(retrieved_context or []),
        retrieved_abox_context=_retrieved_abox_context_payload(retrieved_abox_context or []),
        prefix_declarations=prefix_declarations(ontology_context or {}),
        prompt_rules=build_prompt_rules(
            ontology_context or {},
            use_adaptive_naming=use_adaptive_naming,
            use_full_rules=use_full_rules,
        ),
        output_format_instructions=CORRECTION_OUTPUT_FORMAT_INSTRUCTIONS,
    )


def prefix_declarations(ontology_context: dict[str, object]) -> list[str]:
    """Build SPARQL `PREFIX` declarations from ontology context prefixes."""
    prefixes = ontology_context.get("prefixes", [])
    if not isinstance(prefixes, list):
        return []

    declarations: list[str] = []
    for item in prefixes:
        if not isinstance(item, dict):
            continue
        prefix = item.get("prefix")
        namespace = item.get("namespace")
        if not isinstance(prefix, str) or not isinstance(namespace, str):
            continue
        if prefix == ":":
            declarations.append(f"PREFIX : <{namespace}>")
        else:
            declarations.append(f"PREFIX {prefix}: <{namespace}>")
    return declarations


def _retrieved_context_payload(
    retrieved_context: list[RetrievedChunk] | list[dict[str, object]],
) -> list[dict[str, object]]:
    payload = []
    for index, item in enumerate(retrieved_context, 1):
        if isinstance(item, RetrievedChunk):
            payload.append({"rank": item.rank, "text": item.text})
        elif isinstance(item, dict):
            payload.append(
                {
                    "rank": item.get("rank", index),
                    "text": item.get("text", ""),
                }
            )
    return payload


def _retrieved_abox_context_payload(
    retrieved_context: list[RetrievedABoxChunk] | list[dict[str, object]],
) -> list[dict[str, object]]:
    payload = []
    for index, item in enumerate(retrieved_context, 1):
        if isinstance(item, RetrievedABoxChunk):
            payload.append(
                {
                    "rank": item.rank,
                    "uri": item.uri,
                    "display_name": item.display_name,
                    "types": item.types,
                    "text": item.text,
                }
            )
        elif isinstance(item, dict):
            payload.append(
                {
                    "rank": item.get("rank", index),
                    "uri": item.get("uri", ""),
                    "display_name": item.get("display_name", ""),
                    "types": item.get("types", []),
                    "text": item.get("text", ""),
                }
            )
    return payload


def _template_environment() -> Environment:
    template_dir = Path(__file__).resolve().parent / "templates"
    return Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=StrictUndefined,
    )