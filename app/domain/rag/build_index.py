"""Build RAG retrieval artifacts for an ontology package.

This module builds `indexes/<strategy>/chunks.json` and
`indexes/<strategy>/index.faiss` from an existing `ontology_context.json`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.domain.package import (
    PackageNotFoundError,
    chunks_path,
    index_path,
    index_strategy_dir,
    metadata_path,
    read_json_file,
    resolve_package_dir,
    write_json_file,
    ontology_context_path,
)
from app.domain.rag.chunking import SUPPORTED_CHUNKING_ORDER, build_chunks


_EMBEDDING_MODEL: SentenceTransformer | None = None


@dataclass(frozen=True)
class IndexBuildResult:
    """Artifacts produced for one package retrieval index."""

    package_dir: Path
    chunking: str
    chunks_path: Path
    index_path: Path
    chunk_count: int
    embedding_model: str


def build_index(
    package_dir: str | Path,
    *,
    chunking: str = "class_based",
) -> IndexBuildResult:
    """Build chunks and vector index for one ontology package."""
    root = resolve_package_dir(package_dir)
    ontology_context = read_json_file(ontology_context_path(root))
    metadata = read_json_file(metadata_path(root))

    chunks = build_chunks(ontology_context, chunking)
    if not chunks:
        raise PackageNotFoundError("No chunks could be built from ontology_context.json")

    texts = [str(chunk["text"]) for chunk in chunks]
    vectors = embed_texts(texts)
    index = build_vector_index(vectors)

    chunk_dir = index_strategy_dir(root, chunking)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunks_file = chunks_path(root, chunking)
    chunks_file.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    index_file = index_path(root, chunking)
    faiss.write_index(index, str(index_file))

    files_loaded = metadata.setdefault("files_loaded", [])
    if isinstance(files_loaded, list):
        relative_chunks = chunks_file.relative_to(root).as_posix()
        relative_index = index_file.relative_to(root).as_posix()
        if relative_chunks not in files_loaded:
            files_loaded.append(relative_chunks)
        if relative_index not in files_loaded:
            files_loaded.append(relative_index)
    runtime_artifacts = metadata.setdefault("runtime_artifacts", {})
    if not isinstance(runtime_artifacts, dict):
        runtime_artifacts = {}
    indexes = runtime_artifacts.setdefault("indexes", {})
    if not isinstance(indexes, dict):
        indexes = {}
    indexes[chunking] = {
        "chunking_strategy": chunking,
        "chunks_file": chunks_file.relative_to(root).as_posix(),
        "index_file": index_file.relative_to(root).as_posix(),
        "count": len(chunks),
        "embedding_model": settings.rag_embedding_model_name,
    }
    runtime_artifacts["indexes"] = indexes
    metadata["runtime_artifacts"] = runtime_artifacts
    write_json_file(metadata_path(root), metadata)

    return IndexBuildResult(
        package_dir=root,
        chunking=chunking,
        chunks_path=chunks_file,
        index_path=index_file,
        chunk_count=len(chunks),
        embedding_model=settings.rag_embedding_model_name,
    )


def build_all_indexes(package_dir: str | Path) -> list[IndexBuildResult]:
    """Build every supported chunking strategy for one ontology package."""
    return [
        build_index(package_dir, chunking=chunking)
        for chunking in SUPPORTED_CHUNKING_ORDER
    ]


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed chunk or query text with the configured sentence-transformer model."""
    embeddings = embedding_model().encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype="float32")


def build_vector_index(vectors: np.ndarray) -> faiss.Index:
    """Build a simple flat FAISS index."""
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("Expected a non-empty 2D array of vectors")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index


def embedding_model() -> SentenceTransformer:
    """Return a cached sentence-transformer instance."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = SentenceTransformer(settings.rag_embedding_model_name)
    return _EMBEDDING_MODEL
