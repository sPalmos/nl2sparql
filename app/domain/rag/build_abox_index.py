"""Build ABox retrieval artifacts for an ontology package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import faiss
from rdflib import Graph

from app.core.config import settings
from app.domain.package import (
    PackageNotFoundError,
    abox_chunks_path,
    abox_index_dir,
    abox_index_path,
    metadata_path,
    ontology_context_path,
    read_json_file,
    resolve_package_dir,
    write_json_file,
)
from app.domain.rag.abox_chunking import build_abox_chunks
from app.domain.rag.build_index import build_vector_index, embed_texts


@dataclass(frozen=True)
class ABoxIndexBuildResult:
    """Artifacts produced for one package ABox retrieval index."""

    package_dir: Path
    chunks_path: Path
    index_path: Path
    chunk_count: int
    embedding_model: str


def build_abox_index(
    package_dir: str | Path,
    graph: Graph,
) -> ABoxIndexBuildResult:
    """Build ABox chunks and a vector index for one ontology package."""
    root = resolve_package_dir(package_dir)
    ontology_context = read_json_file(ontology_context_path(root))
    metadata = read_json_file(metadata_path(root))

    chunks = build_abox_chunks(graph, ontology_context)
    if not chunks:
        raise PackageNotFoundError("No ABox chunks could be built from the ontology graph")

    texts = [str(chunk["text"]) for chunk in chunks]
    vectors = embed_texts(texts)
    index = build_vector_index(vectors)

    abox_dir = abox_index_dir(root)
    abox_dir.mkdir(parents=True, exist_ok=True)
    chunks_file = abox_chunks_path(root)
    chunks_file.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    index_file = abox_index_path(root)
    faiss.write_index(index, str(index_file))

    files_loaded = metadata.setdefault("files_loaded", [])
    if isinstance(files_loaded, list):
        for artifact in (chunks_file, index_file):
            relative = artifact.relative_to(root).as_posix()
            if relative not in files_loaded:
                files_loaded.append(relative)

    runtime_artifacts = metadata.setdefault("runtime_artifacts", {})
    if not isinstance(runtime_artifacts, dict):
        runtime_artifacts = {}
    runtime_artifacts["abox_index"] = {
        "enabled": True,
        "chunks_file": chunks_file.relative_to(root).as_posix(),
        "index_file": index_file.relative_to(root).as_posix(),
        "count": len(chunks),
        "embedding_model": settings.rag_embedding_model_name,
    }
    metadata["runtime_artifacts"] = runtime_artifacts
    write_json_file(metadata_path(root), metadata)

    return ABoxIndexBuildResult(
        package_dir=root,
        chunks_path=chunks_file,
        index_path=index_file,
        chunk_count=len(chunks),
        embedding_model=settings.rag_embedding_model_name,
    )
