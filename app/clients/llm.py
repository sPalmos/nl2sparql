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
        "options": _ollama_options(),
    }

    async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
        response = await client.post(llm_api_url, json=payload)
        response.raise_for_status()

    body = response.json()
    _log_ollama(model, "generate", payload["options"], body, prompt)
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
        "options": _ollama_options(),
    }

    async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
        response = await client.post(chat_url, json=payload)
        response.raise_for_status()

    body = response.json()
    _log_ollama(model, "chat", payload["options"], body, _first_user_text(messages))
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
    extra_body = os.environ.get("NL2SPARQL_API_EXTRA_BODY")
    if extra_body:
        # e.g. '{"reasoning_effort": "low"}' (paper-v1.1; unset = paper-v1 request)
        payload.update(json.loads(extra_body))

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
    # Logged before the content check (paper-v1.1), so empty or truncated
    # responses are counted too; finish_reason shows length truncation.
    _log_usage(api_model, body.get("usage") or {},
               finish_reason=choices[0].get("finish_reason") if choices else None)
    content = (choices[0].get("message") or {}).get("content", "") if choices else ""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("The API chat response is missing content")
    return content, messages + [{"role": "assistant", "content": content}]


def _log_usage(model: str, usage: dict, finish_reason: str | None = None) -> None:
    try:
        path = Path("logs") / "api_usage.jsonl"
        path.parent.mkdir(exist_ok=True)
        entry = {"t": time.time(), "model": model, **usage}
        if finish_reason is not None:
            entry["finish_reason"] = finish_reason
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
    except OSError:
        pass


# --------------------------------------------------------------------------
# paper-v1.1: optional overrides and generation metadata
# --------------------------------------------------------------------------
# Environment variables (all optional; unset = the paper-v1 request):
#   NL2SPARQL_TEMPERATURE   Ollama temperature (default settings.llm_temperature)
#   NL2SPARQL_TOP_P         Ollama top_p (default: not sent, model default)
#   NL2SPARQL_NUM_CTX       Ollama num_ctx (default settings.llm_num_ctx)
#   NL2SPARQL_API_EXTRA_BODY JSON merged into hosted-API requests
#   NL2SPARQL_LOG_RAW=1     also store the raw model output in the log
# Every Ollama call appends done_reason, eval_count, prompt_eval_count and
# durations to logs/ollama_usage.jsonl. Logging never changes a result.

_OVERRIDE_VARS = (
    "NL2SPARQL_TEMPERATURE",
    "NL2SPARQL_TOP_P",
    "NL2SPARQL_NUM_CTX",
    "NL2SPARQL_API_EXTRA_BODY",
    "NL2SPARQL_API_NO_TEMPERATURE",
    "NL2SPARQL_API_MAX_TOKENS_PARAM",
    "NL2SPARQL_API_BASE",
)


def active_overrides() -> dict[str, str]:
    """The override variables set for this process (never the API key)."""
    return {name: os.environ[name] for name in _OVERRIDE_VARS if os.environ.get(name)}


def _ollama_options() -> dict[str, object]:
    options: dict[str, object] = {
        "temperature": float(os.environ["NL2SPARQL_TEMPERATURE"])
        if os.environ.get("NL2SPARQL_TEMPERATURE") else settings.llm_temperature,
        "num_ctx": int(os.environ["NL2SPARQL_NUM_CTX"])
        if os.environ.get("NL2SPARQL_NUM_CTX") else settings.llm_num_ctx,
        "num_predict": settings.llm_num_predict,
    }
    if os.environ.get("NL2SPARQL_TOP_P"):
        options["top_p"] = float(os.environ["NL2SPARQL_TOP_P"])
    return options


def _first_user_text(messages: list[dict[str, str]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def _log_ollama(model: str, endpoint: str, options: dict, body: dict, prompt: str) -> None:
    try:
        if endpoint == "chat":
            message = body.get("message") or {}
            text = message.get("content") or ""
            thinking = message.get("thinking") or ""
        else:
            text = body.get("response") or ""
            thinking = body.get("thinking") or ""
        entry = {
            "t": time.time(),
            "model": model,
            "endpoint": endpoint,
            "options": options,
            "done_reason": body.get("done_reason"),
            "prompt_eval_count": body.get("prompt_eval_count"),
            "eval_count": body.get("eval_count"),
            "total_duration": body.get("total_duration"),
            "load_duration": body.get("load_duration"),
            "prompt_eval_duration": body.get("prompt_eval_duration"),
            "eval_duration": body.get("eval_duration"),
            "response_chars": len(text),
            "thinking_chars": len(thinking),
            "prompt_tail": prompt[-300:],
        }
        if os.environ.get("NL2SPARQL_LOG_RAW") == "1":
            entry["response"] = text
            entry["thinking"] = thinking
        path = Path("logs") / "ollama_usage.jsonl"
        path.parent.mkdir(exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass
