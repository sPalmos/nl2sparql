"""Thin LLM transport helper.

The runtime pipeline owns prompt construction and output validation. This
module only sends a prompt to the configured LLM endpoint and returns text.

Two modes:
- `generate_text` — single-turn via /api/generate (legacy)
- `chat_completion` — multi-turn via /api/chat, used by the correction loop so
  the model retains conversational context across correction iterations.

Hosted API models (added for the API-model comparison)
------------------------------------------------------
A model name starting with "api:" is sent to an OpenAI-compatible
Chat Completions endpoint instead of Ollama; everything else is unchanged.
Most providers expose such an endpoint (OpenAI, Google Gemini, Anthropic,
OpenRouter, ...). Configure with environment variables:

  NL2SPARQL_API_BASE        base URL, e.g. https://api.openai.com/v1
  NL2SPARQL_API_KEY         the API key
  NL2SPARQL_API_MAX_TOKENS_PARAM   "max_tokens" (default) or "max_completion_tokens"
  NL2SPARQL_API_NO_TEMPERATURE     set to 1 for models that reject `temperature`

Example: --model api:<provider-model-name>

Each API call appends its token usage to logs/api_usage.jsonl (for the cost
report). Rate-limit and server errors are retried with exponential backoff.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import httpx

from app.core.config import settings

API_PREFIX = "api:"
_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


async def generate_text(prompt: str, *, model: str, llm_api_url: str) -> str:
    """Send one prompt to the configured LLM endpoint and return raw text."""
    if model.startswith(API_PREFIX):
        text, _ = await _api_chat([{"role": "user", "content": prompt}], model)
        return text

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": settings.llm_temperature,
            "num_ctx": settings.llm_num_ctx,
            "num_predict": settings.llm_num_predict,
        },
    }

    async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
        response = await client.post(llm_api_url, json=payload)
        response.raise_for_status()

    body = response.json()
    generated = body.get("response")
    if not isinstance(generated, str):
        raise ValueError("The LLM response is missing generated text")
    return generated


async def chat_completion(
    messages: list[dict[str, str]],
    *,
    model: str,
    llm_api_url: str,
) -> tuple[str, list[dict[str, str]]]:
    """Send a conversation to the chat endpoint and return (text, updated_messages).

    Uses the Ollama /api/chat endpoint which maintains multi-turn context. The
    returned `updated_messages` includes the new assistant response appended,
    ready for the next correction turn.

    The chat URL is derived from `llm_api_url` by replacing /api/generate with
    /api/chat.
    """
    if model.startswith(API_PREFIX):
        return await _api_chat(messages, model)

    chat_url = _chat_url(llm_api_url)
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": settings.llm_temperature,
            "num_ctx": settings.llm_num_ctx,
            "num_predict": settings.llm_num_predict,
        },
    }

    async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
        response = await client.post(chat_url, json=payload)
        response.raise_for_status()

    body = response.json()
    message = body.get("message", {})
    content = message.get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("The LLM chat response is missing content")

    updated_messages = messages + [{"role": "assistant", "content": content}]
    return content, updated_messages


def _chat_url(llm_api_url: str) -> str:
    """Derive the chat endpoint URL from the generate endpoint URL."""
    if "/api/generate" in llm_api_url:
        return llm_api_url.replace("/api/generate", "/api/chat")
    base = llm_api_url.rstrip("/")
    if base.endswith("/api"):
        return base + "/chat"
    return base + "/api/chat"


# --------------------------------------------------------------------------
# OpenAI-compatible hosted API
# --------------------------------------------------------------------------

async def _api_chat(
    messages: list[dict[str, str]], model: str, *, max_retries: int = 6,
) -> tuple[str, list[dict[str, str]]]:
    base = os.environ.get("NL2SPARQL_API_BASE", "").rstrip("/")
    key = os.environ.get("NL2SPARQL_API_KEY", "")
    if not base or not key:
        raise ValueError("Set NL2SPARQL_API_BASE and NL2SPARQL_API_KEY to use an api: model")

    api_model = model[len(API_PREFIX):]
    payload: dict[str, object] = {"model": api_model, "messages": messages}
    payload[os.environ.get("NL2SPARQL_API_MAX_TOKENS_PARAM", "max_tokens")] = settings.llm_num_predict
    if os.environ.get("NL2SPARQL_API_NO_TEMPERATURE", "0") != "1":
        payload["temperature"] = settings.llm_temperature

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    delay = 2.0
    async with httpx.AsyncClient(timeout=max(settings.llm_timeout_seconds, 180.0)) as client:
        for attempt in range(max_retries + 1):
            try:
                response = await client.post(f"{base}/chat/completions", json=payload, headers=headers)
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt == max_retries:
                    raise
                await asyncio.sleep(delay)
                delay *= 2
                continue
            if response.status_code in _RETRY_STATUS and attempt < max_retries:
                retry_after = response.headers.get("retry-after")
                await asyncio.sleep(float(retry_after) if retry_after and retry_after.isdigit() else delay)
                delay *= 2
                continue
            response.raise_for_status()
            break

    body = response.json()
    choices = body.get("choices") or []
    content = (choices[0].get("message") or {}).get("content", "") if choices else ""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("The API chat response is missing content")

    _log_usage(api_model, body.get("usage") or {})
    return content, messages + [{"role": "assistant", "content": content}]


def _log_usage(model: str, usage: dict) -> None:
    try:
        path = Path("logs") / "api_usage.jsonl"
        path.parent.mkdir(exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"t": time.time(), "model": model, **usage}) + "\n")
    except OSError:
        pass