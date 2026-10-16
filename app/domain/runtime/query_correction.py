"""Generate corrected SPARQL candidates after failed attempts.

Two modes:
- `correct_query` — renders a full correction prompt and calls /api/generate
  (legacy, stateless)
- `correct_query_chat` — appends a correction message to the existing chat
  history and calls /api/chat, so the model sees all previous attempts and
  feedback in one conversation.
"""

from __future__ import annotations

from app.clients.llm import chat_completion, generate_text
from app.domain.rag import RetrievedABoxChunk, RetrievedChunk
from app.domain.runtime.prompt_renderer import render_correction_prompt
from app.domain.runtime.query_generation import normalize_generated_query


async def correct_query(
    *,
    question: str,
    failed_query: str,
    validation_errors: list[str],
    retrieved_context: list[RetrievedChunk],
    retrieved_abox_context: list[RetrievedABoxChunk] | None = None,
    ontology_context: dict[str, object],
    model: str,
    llm_api_url: str,
) -> str:
    """Return the next normalized SPARQL candidate from correction feedback."""
    correction_prompt = render_correction_prompt(
        original_question=question,
        failed_query=failed_query,
        validation_errors=validation_errors,
        retrieved_context=retrieved_context,
        retrieved_abox_context=retrieved_abox_context,
        ontology_context=ontology_context,
    )
    return normalize_generated_query(
        await generate_text(correction_prompt, model=model, llm_api_url=llm_api_url)
    )


async def correct_query_chat(
    *,
    message_history: list[dict[str, str]],
    correction_message: str,
    model: str,
    llm_api_url: str,
) -> tuple[str, list[dict[str, str]]]:
    """Correct a query by continuing the chat conversation.

    Appends the correction feedback as a new user message to the existing
    conversation history, calls the chat API, and returns the corrected query
    plus the updated history (now including the correction request and the
    model's response). The model retains full context of prior attempts.
    """
    messages = message_history + [{"role": "user", "content": correction_message}]
    raw_response, updated_messages = await chat_completion(
        messages, model=model, llm_api_url=llm_api_url
    )
    query = normalize_generated_query(raw_response)
    return query, updated_messages
