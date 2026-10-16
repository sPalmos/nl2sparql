"""Generate and normalize initial SPARQL candidates.

This module owns the first LLM call in the runtime flow. It receives the fully
rendered generation prompt, calls the configured LLM endpoint, and normalizes the
raw model output into plain SPARQL.

Two modes:
- `generate_initial_query` — single-turn via /api/generate (legacy)
- `generate_initial_query_chat` — multi-turn via /api/chat, returns the message
  history so the correction loop can continue the same conversation.

Output normalization is model-agnostic: it strips reasoning-model `<think>`
blocks and fenced code blocks uniformly, so reasoning models (e.g. DeepSeek-R1)
and code models run through the identical pipeline.
"""

from __future__ import annotations

import re

from app.clients.llm import chat_completion, generate_text


# SPARQL queries always begin with one of these keywords (after prefixes).
_QUERY_START = re.compile(r"\b(SELECT|ASK|CONSTRUCT|DESCRIBE)\b", re.IGNORECASE)


async def generate_initial_query(prompt: str, *, model: str, llm_api_url: str) -> str:
    """Return the first normalized SPARQL candidate from an LLM prompt."""
    return normalize_generated_query(
        await generate_text(prompt, model=model, llm_api_url=llm_api_url)
    )


async def generate_initial_query_chat(
    prompt: str,
    *,
    model: str,
    llm_api_url: str,
    system_role: str = "",
) -> tuple[str, list[dict[str, str]]]:
    """Generate initial SPARQL via chat API, returning (query, message_history).

    The message history contains the system message, the user prompt, and the
    assistant response. The correction loop appends to this history to maintain
    full conversational context across iterations.
    """
    messages: list[dict[str, str]] = []
    if system_role:
        messages.append({"role": "system", "content": system_role})
    messages.append({"role": "user", "content": prompt})

    raw_response, updated_messages = await chat_completion(
        messages, model=model, llm_api_url=llm_api_url
    )
    query = normalize_generated_query(raw_response)
    return query, updated_messages


def normalize_generated_query(generated_text: str) -> str:
    """Normalize raw LLM output into a SPARQL query string.

    Applied uniformly to every model. Steps, in order:
    1. Strip reasoning-model `<think>...</think>` blocks. Reasoning models emit
       their chain-of-thought in this wrapper before the answer; it is not part
       of the query. Handles an unclosed `<think>` (output truncated mid-reason)
       by dropping everything up to the last `</think>` if present.
    2. Extract the SPARQL from a fenced code block if one is present anywhere
       (not only at the start), tolerating surrounding text.
    3. Otherwise, slice from the first SPARQL keyword (SELECT/ASK/CONSTRUCT/
       DESCRIBE) so any leading prose is dropped.

    Empty results are treated as generation failures and raise `ValueError`.
    """
    text = generated_text.strip()

    # 1. Remove reasoning <think> blocks (model-agnostic; no-op if absent).
    text = _strip_think_blocks(text)

    # 2. Prefer a fenced code block anywhere in the text.
    fenced = _extract_fenced_block(text)
    if fenced is not None:
        text = fenced
    else:
        # 3. No fence — drop any leading prose before the query keyword.
        match = _QUERY_START.search(text)
        if match:
            # Keep any PREFIX lines that precede the keyword on their own lines.
            text = _include_leading_prefixes(text, match.start())

    text = text.strip()
    if not text:
        raise ValueError("The LLM returned an empty query")
    return text


def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from model output."""
    # Closed blocks (possibly several).
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Unclosed block from truncated output: if a </think> remains without an
    # opening tag, drop everything before it. If a lone <think> remains with no
    # close, drop everything after it (no usable query followed).
    if "</think>" in text.lower():
        idx = text.lower().rfind("</think>")
        text = text[idx + len("</think>"):]
    elif "<think>" in text.lower():
        idx = text.lower().find("<think>")
        text = text[:idx]
    return text.strip()


def _extract_fenced_block(text: str) -> str | None:
    """Return the contents of the first fenced code block, or None if absent."""
    fence = re.search(r"```[A-Za-z0-9_-]*\s*(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return None


def _include_leading_prefixes(text: str, keyword_pos: int) -> str:
    """Include PREFIX declarations that appear on lines before the query keyword.

    When slicing prose off the front, we still want to keep any PREFIX lines that
    directly precede the SELECT/ASK/etc. keyword.
    """
    head = text[:keyword_pos]
    prefix_lines = re.findall(r"(?im)^\s*PREFIX\b.*$", head)
    body = text[keyword_pos:]
    if prefix_lines:
        return "\n".join(prefix_lines) + "\n" + body
    return body