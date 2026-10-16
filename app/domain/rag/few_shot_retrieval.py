"""Retrieve semantically similar few-shot examples for prompt injection.

This module loads (question, SPARQL) example pairs from a JSON file, embeds
the questions using the same sentence-transformer model as the RAG pipeline,
and retrieves the N most similar examples for a given user question.

The few-shot examples file is expected at:
    ontology_packages/<package>/few_shot_examples.json

If the file does not exist, retrieval returns an empty list (graceful fallback).
"""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import numpy as np

# Reuse the same embedding function from the RAG index builder
index_module = import_module("app.domain.rag.build_index")


def retrieve_few_shot_examples(
    package_dir: str | Path,
    question: str,
    *,
    n: int = 3,
) -> list[dict[str, str]]:
    """Return the N most similar few-shot examples for the given question.

    Each returned dict has keys: "question", "sparql".

    Returns an empty list if:
    - The few-shot examples file does not exist
    - The file is empty or malformed
    - n <= 0
    """
    if n <= 0:
        return []

    root = Path(package_dir)
    examples_path = root / "few_shot_examples.json"
    if not examples_path.exists():
        return []

    try:
        examples = json.loads(examples_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    if not isinstance(examples, list) or not examples:
        return []

    # Filter to valid examples
    valid_examples = [
        ex for ex in examples
        if isinstance(ex, dict)
        and isinstance(ex.get("question"), str)
        and isinstance(ex.get("sparql"), str)
    ]
    if not valid_examples:
        return []

    # Embed all example questions + the user question in one batch
    example_questions = [ex["question"] for ex in valid_examples]
    all_texts = example_questions + [question.strip()]
    embeddings = index_module.embed_texts(all_texts)

    # Last embedding is the user question
    query_embedding = embeddings[-1:]
    example_embeddings = embeddings[:-1]

    # Compute cosine similarities
    # Normalize for cosine similarity (embeddings may already be normalized,
    # but this is safe regardless)
    query_norm = query_embedding / (np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-9)
    example_norms = example_embeddings / (np.linalg.norm(example_embeddings, axis=1, keepdims=True) + 1e-9)
    similarities = (example_norms @ query_norm.T).flatten()

    # Get top-N indices by similarity (descending)
    top_n = min(n, len(valid_examples))
    top_indices = np.argsort(similarities)[::-1][:top_n]

    return [
        {
            "question": valid_examples[idx]["question"],
            "sparql": valid_examples[idx]["sparql"],
        }
        for idx in top_indices
    ]
