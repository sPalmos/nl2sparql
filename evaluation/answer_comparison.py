"""Compare generated SPARQL answers with gold answers.

The runtime and the gold query can return the same logical answer with small
surface differences: URI values may be written as full IRIs or prefixed names,
typed literals may include datatype suffixes, numbers may differ only by
formatting, and result variables may have different names. This module
normalizes those cases before scoring.

The public entry point is `compare_results()`. It returns exact match,
precision, recall, F1, and the missing/extra normalized rows used by evaluation
reports. It deliberately does not execute SPARQL or know about datasets; it only
compares already-materialized result rows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


def normalize_uri(value: str, prefix_map: Optional[dict[str, str]] = None) -> str:
    """Return a canonical URI string for full IRIs, angle-bracket IRIs, or prefixed names.

    `prefix_map` maps prefixes such as `ex` to namespaces. When provided,
    `ex:Thing` and `http://example/.../Thing` can compare equal. Unknown
    prefixes are left unchanged so mismatches remain visible in the diff.
    """
    value = value.strip()
    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1]

    if prefix_map and ":" in value and not value.startswith("http"):
        prefix, _, local = value.partition(":")
        if prefix in prefix_map:
            value = prefix_map[prefix] + local
    return value


def normalize_literal(value: str) -> str:
    """Return a canonical literal string for answer comparison.

    This removes common RDF literal wrappers, language tags, and datatype
    suffixes, then normalizes numeric and boolean spellings. The goal is to
    avoid penalizing harmless serialization differences while preserving the
    literal's logical value.
    """
    value = value.strip()

    datatype_match = re.match(r'^"?(.*?)"?\^\^<?[^>]+>?$', value)
    if datatype_match:
        value = datatype_match.group(1)

    lang_match = re.match(r'^"?(.*?)"?@[a-zA-Z\-]+$', value)
    if lang_match:
        value = lang_match.group(1)

    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        value = value[1:-1]

    try:
        number = float(value)
        if number == int(number):
            return str(int(number))
        # Round non-integers to 6 significant digits. Gold answers produced by
        # the original float32 pipelines store values such as 62.9000015 where
        # the graph holds 62.9; comparing at 6 significant digits removes that
        # representation artefact without merging genuinely different numbers.
        return f"{number:.6g}"
    except (ValueError, OverflowError):
        pass

    if value.lower() in ("true", "yes"):
        return "true"
    if value.lower() in ("false", "no"):
        return "false"
    return value.strip()


def normalize_answer_surface(value: str, prefix_map: Optional[dict[str, str]] = None) -> str:
    """Return a lexical answer key that makes URI local names comparable to labels.

    Evaluation datasets sometimes store resource answers as URIs while generated
    queries return the human-readable name, or the reverse. For answer scoring,
    `http://example/Karen_Brant`, `ex:Karen_Brant`, and `Karen Brant` should be
    treated as the same surface answer when their lexical forms agree.

    This is deliberately a scoring normalization only. It does not rewrite the
    generated SPARQL or the raw answers saved in evaluation logs.
    """
    return normalize_answer_surface_with_aliases(value, prefix_map=prefix_map, answer_aliases=None)


def normalize_answer_surface_with_aliases(
    value: str,
    *,
    prefix_map: Optional[dict[str, str]] = None,
    answer_aliases: Optional[dict[str, list[str]]] = None,
) -> str:
    """Return a lexical answer key, resolving URI answers through known aliases."""
    value = normalize_value(value, prefix_map)
    if value.startswith("http://") or value.startswith("https://"):
        aliases = answer_aliases.get(value, []) if answer_aliases else []
        if aliases:
            return _surface_key(aliases[0])
        value = _local_name(value)
    return _surface_key(value)


def normalize_value(value: str, prefix_map: Optional[dict[str, str]] = None) -> str:
    """Normalize one SPARQL result cell as either a URI-like value or a literal."""
    value = value.strip()
    is_uri = (
        value.startswith("http://")
        or value.startswith("https://")
        or (value.startswith("<") and value.endswith(">"))
    )

    if not is_uri and prefix_map and ":" in value:
        is_uri = value.split(":", 1)[0] in prefix_map

    if is_uri:
        return normalize_uri(value, prefix_map)
    return normalize_literal(value)


def normalize_row(
    row: dict[str, str],
    prefix_map: Optional[dict[str, str]] = None,
    answer_aliases: Optional[dict[str, list[str]]] = None,
) -> tuple[str, ...]:
    """Normalize one result row into a sorted tuple of values.

    Variable names are intentionally ignored. For evaluation, a row containing
    `?x = Alice` is treated the same as a row containing `?label = Alice`.
    Values are deduplicated and sorted so column order does not affect equality,
    and returning both a URI and its matching label does not create a spurious
    extra-column mismatch.
    """
    return tuple(
        sorted(
            {
                normalize_answer_surface_with_aliases(
                    str(value),
                    prefix_map=prefix_map,
                    answer_aliases=answer_aliases,
                )
                for value in row.values()
            }
        )
    )


def _local_name(uri: str) -> str:
    if "#" in uri:
        return uri.rsplit("#", 1)[-1]
    return uri.rstrip("/").rsplit("/", 1)[-1]


def _surface_key(value: str) -> str:
    value = value.strip()
    if not value:
        return ""

    # Split compact terms that remain unresolved, e.g. ex:Karen_Brant.
    if re.match(r"^[A-Za-z_][\w-]*:[^/].*$", value):
        value = value.split(":", 1)[1]

    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    value = re.sub(r"[_\-/]+", " ", value)
    value = re.sub(r"[^\w\s.]+", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value).strip().lower()
    return value


def normalize_result_set(
    results: list[dict[str, str]],
    prefix_map: Optional[dict[str, str]] = None,
    answer_aliases: Optional[dict[str, list[str]]] = None,
) -> set[tuple[str, ...]]:
    """Normalize all result rows into a set suitable for exact and partial matching."""
    return {normalize_row(row, prefix_map, answer_aliases) for row in results}


@dataclass
class ComparisonResult:
    """Scoring details for one generated-vs-gold answer comparison.

    `exact_match` is true only when no normalized rows are missing or extra.
    Precision/recall/F1 are computed over normalized row sets. `missing_rows`
    and `extra_rows` are stored for readable evaluation logs.
    """

    exact_match: bool = False
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    gold_size: int = 0
    generated_size: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    gold_is_empty: bool = False
    generated_is_empty: bool = False
    generated_is_none: bool = False

    missing_rows: list[tuple] = field(default_factory=list)
    extra_rows: list[tuple] = field(default_factory=list)


def compare_results(
    generated: Optional[list[dict[str, str]]],
    gold: list[dict[str, str]],
    prefix_map: Optional[dict[str, str]] = None,
    answer_aliases: Optional[dict[str, list[str]]] = None,
) -> ComparisonResult:
    """Compare generated and gold result sets using normalized row-set overlap.

    `generated=None` represents a pipeline failure or non-result, while an empty
    list represents a successful query that returned no rows. Empty gold answers
    are handled explicitly so unscored questions can still be represented
    consistently by the caller.
    """
    result = ComparisonResult(gold_is_empty=len(gold) == 0)

    if generated is None:
        result.generated_is_none = True
        result.generated_is_empty = True
        result.gold_size = len(gold)
        result.false_negatives = len(gold)
        result.precision = 1.0
        result.recall = 0.0
        result.f1 = 0.0
        return result

    result.generated_is_empty = len(generated) == 0

    if result.gold_is_empty and result.generated_is_empty:
        result.exact_match = True
        result.precision = 1.0
        result.recall = 1.0
        result.f1 = 1.0
        return result

    if result.gold_is_empty and not result.generated_is_empty:
        result.generated_size = len(generated)
        result.false_positives = len(generated)
        result.precision = 0.0
        result.recall = 1.0
        result.f1 = 0.0
        return result

    if result.generated_is_empty and not result.gold_is_empty:
        result.gold_size = len(gold)
        result.false_negatives = len(gold)
        result.precision = 1.0
        result.recall = 0.0
        result.f1 = 0.0
        return result

    gold_set = normalize_result_set(gold, prefix_map, answer_aliases)
    generated_set = normalize_result_set(generated, prefix_map, answer_aliases)

    result.gold_size = len(gold_set)
    result.generated_size = len(generated_set)

    true_positives, false_positives, false_negatives = _match_rows(gold_set, generated_set)

    result.true_positives = true_positives
    result.false_positives = len(false_positives)
    result.false_negatives = len(false_negatives)
    result.missing_rows = sorted(false_negatives)
    result.extra_rows = sorted(false_positives)
    result.exact_match = not false_positives and not false_negatives

    result.precision = true_positives / len(generated_set) if generated_set else 0.0
    result.recall = true_positives / len(gold_set) if gold_set else 0.0
    if result.precision + result.recall > 0:
        result.f1 = 2 * result.precision * result.recall / (result.precision + result.recall)
    return result


def _match_rows(
    gold_set: set[tuple[str, ...]],
    generated_set: set[tuple[str, ...]],
) -> tuple[int, set[tuple[str, ...]], set[tuple[str, ...]]]:
    """Match rows, allowing generated rows to contain extra helper values.

    Exact equality is preferred. After that, a generated row can match a gold row
    when it is a strict superset of the gold row. This handles generated answers
    that SELECT both an entity URI and its label while gold stores only the
    answer label. The reverse is not accepted because fewer generated values
    means part of the expected answer is missing.
    """
    unmatched_gold = set(gold_set)
    unmatched_generated = set(generated_set)

    exact_matches = unmatched_gold & unmatched_generated
    unmatched_gold -= exact_matches
    unmatched_generated -= exact_matches
    true_positives = len(exact_matches)

    subset_matched_gold: set[tuple[str, ...]] = set()
    for gold_row in sorted(unmatched_gold):
        gold_values = set(gold_row)
        match = next(
            (
                generated_row
                for generated_row in sorted(unmatched_generated)
                if gold_values < set(generated_row)
            ),
            None,
        )
        if match is None:
            continue
        subset_matched_gold.add(gold_row)
        unmatched_generated.remove(match)
        true_positives += 1

    unmatched_gold -= subset_matched_gold
    return true_positives, unmatched_generated, unmatched_gold