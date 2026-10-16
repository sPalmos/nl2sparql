"""Retrieve relevant text chunks from a package RAG index."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

import faiss

from app.core.config import settings
from app.domain.package import chunks_path, index_path, read_json_list, resolve_package_dir

index_module = import_module("app.domain.rag.build_index")


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieved chunk result."""

    rank: int
    score: float
    class_name: str | None
    class_uri: str | None
    text: str | None
    metadata: dict[str, object] | None

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "score": self.score,
            "class_name": self.class_name,
            "class_uri": self.class_uri,
            "text": self.text,
            "metadata": self.metadata,
        }


def retrieve_context(
    package_dir: str | Path,
    question: str,
    *,
    k: int = 5,
    chunking: str | None = None,
) -> list[RetrievedChunk]:
    """Return the top-k retrieved chunks for one question."""
    root = resolve_package_dir(package_dir)
    effective_k = max(1, k or settings.runtime_retrieval_top_k)
    effective_chunking = chunking or settings.default_chunking_strategy

    chunks = read_json_list(chunks_path(root, effective_chunking))
    index = faiss.read_index(str(index_path(root, effective_chunking)))
    query_vector = index_module.embed_texts([question.strip()])
    search_k = min(effective_k, index.ntotal)
    distances, indices = index.search(query_vector, k=search_k)

    results: list[RetrievedChunk] = []
    for rank, chunk_index in enumerate(indices[0], start=1):
        if chunk_index < 0 or chunk_index >= len(chunks):
            continue
        chunk = chunks[chunk_index]
        metadata = chunk.get("metadata")
        results.append(
            RetrievedChunk(
                rank=rank,
                score=float(distances[0][rank - 1]),
                class_name=chunk.get("class_name") if isinstance(chunk.get("class_name"), str) else None,
                class_uri=chunk.get("class_uri") if isinstance(chunk.get("class_uri"), str) else None,
                text=chunk.get("text") if isinstance(chunk.get("text"), str) else None,
                metadata=metadata if isinstance(metadata, dict) else None,
            )
        )
    return results


def retrieve_text_chunks(
    package_dir: str | Path,
    question: str,
    *,
    k: int = 5,
    chunking: str | None = None,
) -> list[str]:
    """Return only retrieved chunk text for simple callers."""
    return [
        chunk.text
        for chunk in retrieve_context(package_dir, question, k=k, chunking=chunking)
        if isinstance(chunk.text, str)
    ]

