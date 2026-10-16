"""Retrieve relevant instance-level chunks from a package ABox index."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

import faiss

from app.core.config import settings
from app.domain.package import abox_chunks_path, abox_index_path, read_json_list, resolve_package_dir

index_module = import_module("app.domain.rag.build_index")


@dataclass(frozen=True)
class RetrievedABoxChunk:
    """One retrieved instance-level chunk result."""

    rank: int
    score: float
    uri: str | None
    display_name: str | None
    types: list[str]
    text: str | None
    metadata: dict[str, object] | None

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "score": self.score,
            "uri": self.uri,
            "display_name": self.display_name,
            "types": self.types,
            "text": self.text,
            "metadata": self.metadata,
        }


def retrieve_abox_context(
    package_dir: str | Path,
    question: str,
    *,
    k: int | None = None,
    use_rerank: bool = True,
    use_gating: bool = True,
) -> list[RetrievedABoxChunk]:
    """Return top-k retrieved ABox chunks, or an empty list if no ABox index exists.

    `use_rerank=False` keeps the raw FAISS distance order instead of promoting
    candidates whose names appear in the question. `use_gating=False` always
    returns candidates instead of suppressing weak generic matches. The two are
    separate switches so their contributions can be measured independently.
    """
    root = resolve_package_dir(package_dir)
    chunks_file = abox_chunks_path(root)
    index_file = abox_index_path(root)
    if not chunks_file.exists() or not index_file.exists():
        return []

    effective_k = max(1, k or settings.runtime_abox_retrieval_top_k)
    chunks = read_json_list(chunks_file)
    index = faiss.read_index(str(index_file))
    query_vector = index_module.embed_texts([question.strip()])
    search_k = min(max(effective_k * 4, 20), index.ntotal)
    distances, indices = index.search(query_vector, k=search_k)

    candidates: list[tuple[float, float, dict[str, object], bool]] = []
    for rank, chunk_index in enumerate(indices[0], start=1):
        if chunk_index < 0 or chunk_index >= len(chunks):
            continue
        chunk = chunks[chunk_index]
        distance = float(distances[0][rank - 1])
        name_match = _has_name_match(question, chunk)
        score = _rerank_score(question, chunk, distance) if use_rerank else distance
        candidates.append((score, distance, chunk, name_match))

    ranked = sorted(candidates, key=lambda item: item[0])
    if use_gating and not _should_include_abox_candidates(ranked):
        return []

    results: list[RetrievedABoxChunk] = []
    for rank, (_, distance, chunk, _) in enumerate(ranked[:effective_k], start=1):
        metadata = chunk.get("metadata")
        types = chunk.get("types")
        results.append(
            RetrievedABoxChunk(
                rank=rank,
                score=distance,
                uri=chunk.get("uri") if isinstance(chunk.get("uri"), str) else None,
                display_name=chunk.get("display_name") if isinstance(chunk.get("display_name"), str) else None,
                types=[str(item) for item in types if isinstance(item, str)] if isinstance(types, list) else [],
                text=chunk.get("text") if isinstance(chunk.get("text"), str) else None,
                metadata=metadata if isinstance(metadata, dict) else None,
            )
        )
    return results


def _should_include_abox_candidates(
    ranked: list[tuple[float, float, dict[str, object], bool]],
) -> bool:
    """Suppress ABox context when retrieval found only weak generic examples."""
    if not ranked:
        return False
    best_score, best_distance, _, best_name_match = ranked[0]
    return best_name_match or best_score <= 1.0 or best_distance <= 1.0


def _rerank_score(question: str, chunk: dict[str, object], distance: float) -> float:
    """Boost chunks whose concrete names are explicitly mentioned in the question."""
    question_text = question.lower()
    names = _candidate_names(chunk)
    score = distance
    for name in names:
        normalized = name.lower()
        if len(normalized) < 3:
            continue
        if normalized in question_text:
            score -= 1.0
        elif _token_overlap(normalized, question_text) >= 0.75:
            score -= 0.4
    return score


def _has_name_match(question: str, chunk: dict[str, object]) -> bool:
    question_text = question.lower()
    return any(_name_match_strength(name, question_text) for name in _candidate_names(chunk))


def _name_match_strength(name: str, question_text: str) -> bool:
    normalized = name.lower()
    if len(normalized) < 3:
        return False
    return normalized in question_text or _token_overlap(normalized, question_text) >= 0.75


def _candidate_names(chunk: dict[str, object]) -> list[str]:
    values: list[str] = []
    display_name = chunk.get("display_name")
    if isinstance(display_name, str):
        values.append(display_name)
    local_name = chunk.get("local_name")
    if isinstance(local_name, str):
        values.append(local_name.replace("_", " "))
    names = chunk.get("names")
    if isinstance(names, list):
        for item in names:
            if isinstance(item, dict) and isinstance(item.get("value"), str):
                values.append(item["value"])
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def _token_overlap(candidate: str, question: str) -> float:
    tokens = [token for token in _tokens(candidate) if len(token) > 2]
    if not tokens:
        return 0.0
    matched = sum(1 for token in tokens if token in question)
    return matched / len(tokens)


def _tokens(value: str) -> list[str]:
    normalized = "".join(char.lower() if char.isalnum() else " " for char in value)
    return [token for token in normalized.split() if token]