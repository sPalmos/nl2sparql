"""RAG artifact building and retrieval APIs."""

from app.domain.rag.build_abox_index import ABoxIndexBuildResult, build_abox_index
from app.domain.rag.build_index import IndexBuildResult, build_all_indexes, build_index
from app.domain.rag.chunking import SUPPORTED_CHUNKING_ORDER, SUPPORTED_CHUNKING_STRATEGIES, build_chunks
from app.domain.rag.retrieve_abox_context import RetrievedABoxChunk, retrieve_abox_context
from app.domain.rag.retrieve_context import RetrievedChunk, retrieve_context, retrieve_text_chunks

__all__ = [
    "ABoxIndexBuildResult",
    "IndexBuildResult",
    "RetrievedABoxChunk",
    "RetrievedChunk",
    "SUPPORTED_CHUNKING_STRATEGIES",
    "SUPPORTED_CHUNKING_ORDER",
    "build_abox_index",
    "build_all_indexes",
    "build_index",
    "build_chunks",
    "retrieve_abox_context",
    "retrieve_context",
    "retrieve_text_chunks",
]
