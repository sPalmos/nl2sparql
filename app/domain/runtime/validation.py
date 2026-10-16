"""Formal validation stages for generated SPARQL.

This module validates a candidate query before endpoint execution. It normalizes
missing ontology prefixes, checks SPARQL parser syntax, verifies prefixes and
ontology vocabulary references against `ontology_context.json`, and applies a
small set of structural checks. It does not call the LLM, render prompts, or
execute queries against a SPARQL endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
import re

from rdflib.plugins.sparql.parser import parseQuery


@dataclass(frozen=True)
class ValidationStageResult:
    """Result for one named validation stage.

    `code` is intended for stable programmatic traces, while `message` is a
    human-readable explanation included only when the stage fails.
    """

    stage: str
    passed: bool
    code: str
    message: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "passed": self.passed,
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True)
class QueryValidationResult:
    """Validation outcome for one candidate SPARQL query.

    `normalized_query` includes ontology prefixes injected from the package
    context when they were missing from the generated query. `errors` is the
    ordered list of failed-stage messages/codes passed to correction prompts.
    """

    is_valid: bool
    errors: list[str]
    normalized_query: str
    stages: list[ValidationStageResult]

    def to_dict(self) -> dict[str, object]:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "normalized_query": self.normalized_query,
            "stages": [stage.to_dict() for stage in self.stages],
        }


def validate_query(
    query: str,
    *,
    ontology_context: dict[str, object] | None = None,
    use_vocabulary_validation: bool = True,
) -> QueryValidationResult:
    """Run all formal validation stages for one candidate query.

    The stages are intentionally deterministic and local: syntax parsing,
    prefix availability, vocabulary membership, and structural sanity checks.
    Endpoint execution is handled later by the runtime pipeline.

    `use_vocabulary_validation=False` removes only the vocabulary-membership
    stage, leaving the other three intact. This exists so the contribution of
    vocabulary checking can be measured in isolation.
    """
    context = ontology_context or {}
    normalized_query = _normalized_query(query, context)
    stages = [
        _syntactic_validation(normalized_query),
        _prefix_validation(normalized_query, context),
    ]
    if use_vocabulary_validation:
        stages.append(_vocabulary_validation(normalized_query, context))
    stages.append(_structural_validation(normalized_query, context))
    errors = [stage.message or stage.code for stage in stages if not stage.passed]
    return QueryValidationResult(
        is_valid=not errors,
        errors=errors,
        normalized_query=normalized_query,
        stages=stages,
    )


def _syntactic_validation(query: str) -> ValidationStageResult:
    if not query:
        return _fail("syntactic", "SPARQL_EMPTY", "Generated query is empty")
    try:
        parseQuery(query)
    except Exception as exc:
        return _fail("syntactic", "SPARQL_PARSE_ERROR", f"SPARQL parser error: {exc}")
    return _pass("syntactic", "SPARQL_PARSE_OK")


def _prefix_validation(query: str, ontology_context: dict[str, object]) -> ValidationStageResult:
    declared = _declared_prefixes(query)
    available = _ontology_prefixes(ontology_context)
    used = _used_prefixes(query)
    missing = sorted(prefix for prefix in used if prefix not in declared and prefix not in available)
    if missing:
        return _fail(
            "prefix",
            "PREFIX_UNKNOWN",
            f"Query uses undeclared prefixes: {', '.join(missing)}",
        )
    return _pass("prefix", "PREFIXES_OK")


def _vocabulary_validation(query: str, ontology_context: dict[str, object]) -> ValidationStageResult:
    class_uris, property_uris = _ontology_vocabulary(ontology_context)
    prefix_map = {**_ontology_prefixes(ontology_context), **_declared_prefixes(query)}
    unknown_properties: list[str] = []
    unknown_classes: list[str] = []

    for predicate_uri in _predicate_uris(query, prefix_map):
        if _is_builtin_predicate(predicate_uri):
            continue
        if predicate_uri not in property_uris:
            unknown_properties.append(predicate_uri)

    for class_uri in _rdf_type_object_uris(query, prefix_map):
        if class_uri not in class_uris:
            unknown_classes.append(class_uri)

    all_unknown = unknown_properties + unknown_classes
    if not all_unknown:
        return _pass("vocabulary", "VOCABULARY_OK")

    # Build vocabulary index for fuzzy-match suggestions
    property_index = _build_vocabulary_index(ontology_context, "properties")
    class_index = _build_vocabulary_index(ontology_context, "classes")

    suggestion_lines: list[str] = []

    for uri in sorted(set(unknown_properties)):
        local_name = _local_name(uri)
        suggestions = _suggest_corrections(local_name, property_index)
        if suggestions:
            formatted = "; ".join(
                f"'{s['local_name']}' ({s['uri']}; {s['detail']})" for s in suggestions
            )
            suggestion_lines.append(
                f"Unknown property '{local_name}' — did you mean: {formatted}?"
            )
        else:
            suggestion_lines.append(
                f"Unknown property '{local_name}' — no close match found in ontology."
            )

    for uri in sorted(set(unknown_classes)):
        local_name = _local_name(uri)
        suggestions = _suggest_corrections(local_name, class_index)
        if suggestions:
            formatted = "; ".join(
                f"'{s['local_name']}' ({s['uri']}; {s['detail']})" for s in suggestions
            )
            suggestion_lines.append(
                f"Unknown class '{local_name}' — did you mean: {formatted}?"
            )
        else:
            suggestion_lines.append(
                f"Unknown class '{local_name}' — no close match found in ontology."
            )

    message = " | ".join(suggestion_lines)
    return _fail("vocabulary", "VOCABULARY_UNKNOWN_URI", message)


def _structural_validation(query: str, ontology_context: dict[str, object]) -> ValidationStageResult:
    del ontology_context
    body = _query_body(query)
    upper_body = body.upper()

    if not upper_body.startswith(("SELECT", "ASK", "CONSTRUCT", "DESCRIBE")):
        return _fail(
            "structural",
            "QUERY_FORM_INVALID",
            "Generated query must start with SELECT, ASK, CONSTRUCT, or DESCRIBE",
        )

    if "WHERE" not in upper_body:
        return _fail("structural", "WHERE_MISSING", "Generated query must contain a WHERE clause")

    if "{" not in body or "}" not in body:
        return _fail(
            "structural",
            "WHERE_PATTERN_MISSING",
            "Generated query must contain a graph pattern enclosed in braces",
        )

    select_vars = _select_variables(body)
    where_vars = _where_variables(body)

    aggregate_aliases = _aggregate_aliases(body)
    aggregate_input_vars = _aggregate_input_variables(body)
    normal_select_vars = sorted(var for var in select_vars if var not in aggregate_aliases)

    missing_normal_vars = sorted(var for var in normal_select_vars if var not in where_vars)
    if missing_normal_vars:
        return _fail(
            "structural",
            "SELECT_VARIABLE_NOT_BOUND",
            f"SELECT variables are not bound in WHERE: {', '.join(missing_normal_vars)}",
        )

    missing_aggregate_input_vars = sorted(var for var in aggregate_input_vars if var not in where_vars)
    if missing_aggregate_input_vars:
        return _fail(
            "structural",
            "AGGREGATE_VARIABLE_NOT_BOUND",
            f"Aggregate variables are not bound in WHERE: {', '.join(missing_aggregate_input_vars)}",
        )

    if _has_aggregation(body):
        grouped_vars = _group_by_variables(body)
        ungrouped_vars = sorted(var for var in normal_select_vars if var not in grouped_vars)

        if normal_select_vars and "GROUP BY" not in upper_body:
            return _fail(
                "structural",
                "GROUP_BY_MISSING",
                "GROUP BY must be present when aggregation is used with non-aggregate SELECT variables",
            )

        if ungrouped_vars:
            return _fail(
                "structural",
                "GROUP_BY_VARIABLE_MISSING",
                f"Non-aggregate SELECT variables must appear in GROUP BY: {', '.join(ungrouped_vars)}",
            )

    return _pass("structural", "STRUCTURE_OK")


def _normalized_query(query: str, ontology_context: dict[str, object]) -> str:
    query_body = (query or "").strip()
    if not query_body:
        return ""

    declarations = []
    declared = _declared_prefixes(query_body)
    for prefix, namespace in _ontology_prefixes(ontology_context).items():
        if prefix in declared:
            continue
        declarations.append(f"PREFIX : <{namespace}>" if prefix == "" else f"PREFIX {prefix}: <{namespace}>")

    if declarations:
        query_body = "\n".join([*declarations, "", query_body])

    # Fix bare ontology names missing their ':' prefix
    query_body = _fix_bare_ontology_names(query_body, ontology_context)

    return query_body


def _fix_bare_ontology_names(query: str, ontology_context: dict[str, object]) -> str:
    """Prepend ':' to bare ontology property/class names that are missing their prefix.

    LLMs sometimes generate `hasFacility` instead of `:hasFacility`. This function
    detects known ontology local names appearing as bare words (not preceded by a
    prefix like `:`, `rdf:`, etc.) and adds the default namespace prefix.

    Only names that exactly match a known property or class local name are fixed.
    SPARQL keywords, variables, and already-prefixed terms are left untouched.
    """
    known = _known_local_names(ontology_context)
    if not known:
        return query

    # Build alternation from known names, longest first to prevent partial matches
    sorted_names = sorted(known, key=len, reverse=True)
    names_pattern = "|".join(re.escape(name) for name in sorted_names)

    # Match bare names that appear:
    #   - after whitespace, semicolon, open brace/paren, or comma (but NOT after : ? $ < / # ")
    #   - before whitespace, semicolon, period, close brace/paren, comma, or question mark
    pattern = re.compile(
        r"(?<=[\s;{(,])(" + names_pattern + r")(?=[\s;.},)?])"
    )
    return pattern.sub(r":\1", query)


def _known_local_names(ontology_context: dict[str, object]) -> set[str]:
    """Collect all known property and class local names from the ontology context."""
    names: set[str] = set()
    for section in ("object_properties", "datatype_properties", "classes"):
        entries = ontology_context.get(section, [])
        if not isinstance(entries, list):
            continue
        for item in entries:
            if not isinstance(item, dict):
                continue
            uri = item.get("uri")
            if isinstance(uri, str) and uri:
                local = _local_name(uri)
                # Skip very short names that could collide with SPARQL syntax
                if len(local) > 2:
                    names.add(local)
    return names


def _query_body(query: str) -> str:
    lines = query.splitlines()
    body_lines = []
    skipping_prefix_block = True

    for line in lines:
        stripped = line.strip()
        if skipping_prefix_block and stripped.upper().startswith("PREFIX "):
            continue
        if skipping_prefix_block and not stripped:
            continue
        skipping_prefix_block = False
        body_lines.append(line)

    return "\n".join(body_lines).strip()


def _declared_prefixes(query: str) -> dict[str, str]:
    prefixes: dict[str, str] = {}
    for match in re.finditer(r"(?im)^\s*PREFIX\s+([A-Za-z_][\w-]*|):\s*<([^>]+)>", query):
        prefixes[match.group(1)] = match.group(2)
    return prefixes


def _ontology_prefixes(ontology_context: dict[str, object]) -> dict[str, str]:
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


def _used_prefixes(query: str) -> set[str]:
    query_without_prefixes = "\n".join(
        line for line in query.splitlines() if not line.strip().upper().startswith("PREFIX ")
    )
    used = set()
    for match in re.finditer(r"(?<![A-Za-z0-9_/-])([A-Za-z_][\w-]*):([A-Za-z_][\w.-]*)", query_without_prefixes):
        prefix = match.group(1)
        if prefix.lower() in {"http", "https"}:
            continue
        used.add(prefix)
    return used


def _ontology_vocabulary(ontology_context: dict[str, object]) -> tuple[set[str], set[str]]:
    class_uris = _uris_from_entries(ontology_context.get("classes", []))
    property_uris = _uris_from_entries(ontology_context.get("object_properties", []))
    property_uris.update(_uris_from_entries(ontology_context.get("datatype_properties", [])))
    return class_uris, property_uris


def _uris_from_entries(entries: object) -> set[str]:
    if not isinstance(entries, list):
        return set()
    uris = set()
    for item in entries:
        if isinstance(item, dict) and isinstance(item.get("uri"), str):
            uris.add(str(item["uri"]))
    return uris


def _local_names_from_uris(uris: set[str]) -> set[str]:
    """Extract local names from a set of full URIs for fallback matching.

    When the LLM declares a custom prefix with the wrong namespace, the
    expanded URI won't match the stored URI even though the local name is
    correct. This set enables local-name-only matching as a fallback.
    """
    return {_local_name(uri) for uri in uris if uri}


# ── Fuzzy-match suggestion helpers ────────────────────────────────────────────


def _local_name(uri: str) -> str:
    """Extract the local name from a full URI (after # or last /)."""
    if "#" in uri:
        return uri.split("#")[-1]
    return uri.rstrip("/").rsplit("/", 1)[-1]


def _build_vocabulary_index(
    ontology_context: dict[str, object],
    kind: str,
) -> dict[str, dict[str, str]]:
    """Build a local-name → {uri, local_name, detail} index for fuzzy matching.

    `kind` is either "properties" or "classes".

    For properties, `detail` includes domain → range when available.
    For classes, `detail` includes the class label.
    """
    index: dict[str, dict[str, str]] = {}

    if kind == "properties":
        for section_key in ("object_properties", "datatype_properties"):
            entries = ontology_context.get(section_key, [])
            if not isinstance(entries, list):
                continue
            for item in entries:
                if not isinstance(item, dict):
                    continue
                uri = item.get("uri")
                if not isinstance(uri, str):
                    continue
                local = _local_name(uri)
                # Build a human-readable detail string with domain/range
                domain = _first_label_or_name(item.get("domains", item.get("domain")))
                range_ = _first_label_or_name(item.get("ranges", item.get("range")))
                if domain and range_:
                    detail = f"{domain} → {range_}"
                elif domain:
                    detail = f"domain: {domain}"
                elif range_:
                    detail = f"range: {range_}"
                else:
                    detail = section_key.replace("_", " ").rstrip("s")
                index[local] = {"uri": uri, "local_name": local, "detail": detail}

    elif kind == "classes":
        entries = ontology_context.get("classes", [])
        if isinstance(entries, list):
            for item in entries:
                if not isinstance(item, dict):
                    continue
                uri = item.get("uri")
                if not isinstance(uri, str):
                    continue
                local = _local_name(uri)
                label = item.get("label", local)
                detail = f"class: {label}" if isinstance(label, str) else f"class: {local}"
                index[local] = {"uri": uri, "local_name": local, "detail": detail}

    return index


def _first_label_or_name(value: object) -> str:
    """Extract a readable name from a domain/range field.

    The field may be a string, a list of strings, a list of dicts with 'label'
    or 'uri' keys, or None.
    """
    if isinstance(value, str):
        return _local_name(value) if "/" in value or "#" in value else value
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, str):
            return _local_name(first) if "/" in first or "#" in first else first
        if isinstance(first, dict):
            label = first.get("label") or first.get("uri")
            if isinstance(label, str):
                return _local_name(label) if "/" in label or "#" in label else label
    return ""


def _suggest_corrections(
    unknown_local_name: str,
    vocabulary_index: dict[str, dict[str, str]],
    max_suggestions: int = 2,
    cutoff: float = 0.45,
) -> list[dict[str, str]]:
    """Find the closest matching vocabulary entries for an unknown local name.

    Uses difflib.get_close_matches on the local names. The cutoff of 0.45 is
    intentionally lenient — it's better to suggest a wrong match (the LLM can
    evaluate it) than to miss a correct one.
    """
    known_names = list(vocabulary_index.keys())
    if not known_names:
        return []

    # Try exact case-insensitive match first
    lower_map = {k.lower(): k for k in known_names}
    if unknown_local_name.lower() in lower_map:
        key = lower_map[unknown_local_name.lower()]
        return [vocabulary_index[key]]

    # Fuzzy match on local names
    matches = get_close_matches(
        unknown_local_name,
        known_names,
        n=max_suggestions,
        cutoff=cutoff,
    )

    if not matches:
        # Try case-insensitive fuzzy match as fallback
        lower_known = {k.lower(): k for k in known_names}
        lower_matches = get_close_matches(
            unknown_local_name.lower(),
            list(lower_known.keys()),
            n=max_suggestions,
            cutoff=cutoff,
        )
        matches = [lower_known[m] for m in lower_matches]

    return [vocabulary_index[m] for m in matches if m in vocabulary_index]


def _predicate_uris(query: str, prefix_map: dict[str, str]) -> set[str]:
    body = _where_body(query)
    uris: set[str] = set()
    for statement in re.split(r"\s+\.\s+", body):
        tokens = statement.strip().split()
        if len(tokens) < 3:
            continue
        predicate = tokens[1]
        expanded = _expand_term(predicate, prefix_map)
        if expanded:
            uris.add(expanded)
    return uris


def _rdf_type_object_uris(query: str, prefix_map: dict[str, str]) -> set[str]:
    body = _where_body(query)
    uris: set[str] = set()
    for statement in re.split(r"\s+\.\s+", body):
        tokens = statement.strip().split()
        if len(tokens) < 3:
            continue
        predicate = _expand_term(tokens[1], prefix_map)
        if not _is_rdf_type_predicate(predicate or tokens[1]):
            continue
        expanded = _expand_term(tokens[2], prefix_map)
        if expanded:
            uris.add(expanded)
    return uris


def _where_body(query: str) -> str:
    match = re.search(r"\{(.*)\}", query, re.DOTALL)
    return match.group(1) if match else ""


def _expand_term(term: str, prefix_map: dict[str, str]) -> str | None:
    cleaned = term.strip().strip(";,.")
    if cleaned == "a":
        return "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
    if cleaned.startswith("<") and cleaned.endswith(">"):
        return cleaned[1:-1]
    prefixed = re.match(r"^([A-Za-z_][\w-]*|):([A-Za-z_][\w.-]*)$", cleaned)
    if not prefixed:
        return None
    prefix, local = prefixed.groups()
    namespace = prefix_map.get(prefix)
    return f"{namespace}{local}" if namespace else None


def _is_builtin_predicate(uri: str) -> bool:
    return uri in {
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://www.w3.org/2000/01/rdf-schema#label",
        "http://www.w3.org/2000/01/rdf-schema#comment",
    }


def _is_rdf_type_predicate(value: str) -> bool:
    return value in {
        "a",
        "rdf:type",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
    }


def _select_variables(query_body: str) -> set[str]:
    match = re.search(r"(?is)\bSELECT\b(.*?)\bWHERE\b", query_body)
    if not match:
        return set()
    select_text = re.sub(r"\([^)]*\)", " ", match.group(1))
    if "*" in select_text:
        return set()
    return {var.lstrip("?") for var in re.findall(r"\?[A-Za-z_][\w-]*", select_text)}


def _where_variables(query_body: str) -> set[str]:
    return {var.lstrip("?") for var in re.findall(r"\?[A-Za-z_][\w-]*", _where_body(query_body))}


def _has_aggregation(query_body: str) -> bool:
    return bool(re.search(r"(?i)\b(COUNT|SUM|AVG|MIN|MAX|GROUP_CONCAT|SAMPLE)\s*\(", query_body))


def _aggregate_aliases(query_body: str) -> set[str]:
    pattern = re.compile(
        r"\(\s*(COUNT|SUM|AVG|MIN|MAX|GROUP_CONCAT|SAMPLE)\s*\([^)]*\)\s+AS\s+\?([A-Za-z_][\w-]*)\s*\)",
        re.IGNORECASE,
    )
    return {match.group(2) for match in pattern.finditer(query_body)}


def _aggregate_input_variables(query_body: str) -> set[str]:
    vars_: set[str] = set()
    pattern = re.compile(
        r"\b(COUNT|SUM|AVG|MIN|MAX|GROUP_CONCAT|SAMPLE)\s*\((.*?)\)",
        re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(query_body):
        vars_.update(re.findall(r"\?([A-Za-z_][\w-]*)", match.group(2)))
    return vars_


def _group_by_variables(query_body: str) -> set[str]:
    match = re.search(
        r"\bGROUP\s+BY\b(.*?)(ORDER\s+BY|LIMIT|OFFSET|HAVING|$)",
        query_body,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return set()
    return set(re.findall(r"\?([A-Za-z_][\w-]*)", match.group(1)))


def _pass(stage: str, code: str) -> ValidationStageResult:
    return ValidationStageResult(stage=stage, passed=True, code=code)


def _fail(stage: str, code: str, message: str) -> ValidationStageResult:
    return ValidationStageResult(stage=stage, passed=False, code=code, message=message)