"""Persist runtime query traces."""

from __future__ import annotations

from pathlib import Path
import json


# Upper bound on result rows kept per trace. Traces are diagnostic records, not
# answer stores: scoring uses the in-memory result, never the log. Without a cap a
# single unconstrained query over a large instance graph can write hundreds of
# thousands of rows into the log.
MAX_STORED_ROWS = 100
MAX_RENDERED_ROWS = 50


def write_query_trace(path: Path, payload: dict[str, object]) -> Path:
    """Append one machine-readable runtime trace entry to `query.log`.

    The log is JSON Lines: one self-contained JSON object per line, appended.
    Each write therefore costs the size of one entry, independent of how large
    the log has grown. (An earlier version re-read and rewrote the whole file on
    every question, which made write cost grow with the log; on the largest
    instance graph the log reached several gigabytes and one write took minutes.)

    Result rows are capped at `MAX_STORED_ROWS`, with the true row count kept
    alongside. The payload passed in is not modified, because the same result
    objects are used downstream for scoring.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = _compact(payload)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return path


def read_query_trace(path: Path) -> list[dict[str, object]]:
    """Read every entry from a JSON Lines `query.log`, skipping corrupt lines."""
    if not path.exists():
        return []
    entries: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                entries.append(item)
    return entries


def read_last_query_trace(path: Path) -> dict[str, object] | None:
    """Return the most recent entry without reading the whole log.

    Reads backwards from the end of the file until it has the final line, so
    the cost is the size of one entry, however large the log is. Falls back to
    parsing the whole file only for a log still in the old JSON-array format.
    """
    if not path.exists():
        return None
    with path.open("rb") as handle:
        handle.seek(0, 2)
        end = handle.tell()
        if end == 0:
            return None

        handle.seek(0)
        head = handle.read(64).lstrip()
        if head.startswith(b"["):
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
            if isinstance(loaded, list) and loaded and isinstance(loaded[-1], dict):
                return loaded[-1]
            return None

        block = 65536
        buffer = b""
        position = end
        line = b""
        while position > 0:
            step = min(block, position)
            position -= step
            handle.seek(position)
            buffer = handle.read(step) + buffer
            trimmed = buffer.rstrip()
            newline = trimmed.rfind(b"\n")
            if newline != -1:
                line = trimmed[newline + 1:]
                break
        else:
            line = buffer.strip()

    try:
        item = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return item if isinstance(item, dict) else None


def read_recent_query_traces(path: Path, limit: int = 50) -> list[dict[str, object]]:
    """Return the most recent `limit` entries, oldest first.

    Streams the file line by line and keeps only the tail in memory, so it stays
    cheap however large the log grows. Also accepts a log still in the earlier
    single-JSON-array format, so an un-migrated package keeps working.
    """
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        first = handle.read(1)
        while first and first.isspace():
            first = handle.read(1)
    if first == "[":
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        items = [item for item in loaded if isinstance(item, dict)] if isinstance(loaded, list) else []
        return items[-limit:]

    from collections import deque

    recent: deque[dict[str, object]] = deque(maxlen=limit)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                recent.append(item)
    return list(recent)


def _compact(value: object) -> object:
    """Return a copy of `value` with every SPARQL bindings list capped.

    Builds new containers rather than mutating, so the caller's objects (which
    feed answer scoring) are untouched.
    """
    if isinstance(value, dict):
        copied = {key: _compact(item) for key, item in value.items()}
        results = copied.get("results")
        if isinstance(results, dict):
            bindings = results.get("bindings")
            if isinstance(bindings, list) and len(bindings) > MAX_STORED_ROWS:
                copied["results"] = {
                    **results,
                    "bindings": bindings[:MAX_STORED_ROWS],
                    "total_rows": len(bindings),
                    "truncated": True,
                }
        return copied
    if isinstance(value, list):
        return [_compact(item) for item in value]
    return value


def write_readable_query_trace(
    *,
    latest_path: Path,
    runs_dir: Path,
    run_id: str,
    payload: dict[str, object],
) -> Path:
    """Write a compact plain-text query trace for humans."""
    text = render_readable_query_trace(payload)
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_path = runs_dir / f"{run_id}.txt"
    run_path.write_text(text, encoding="utf-8")
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(text, encoding="utf-8")
    return run_path


def render_readable_query_trace(payload: dict[str, object]) -> str:
    """Render one query trace payload as compact plain text."""
    lines = [
        "QUERY RUN",
        f"Run ID: {_text(payload.get('run_id'))}",
        f"Run At: {_text(payload.get('run_at'))}",
        f"Status: {_text(payload.get('status'))}",
        f"Dataset: {_text(payload.get('dataset_name'))}",
        f"Endpoint: {_text(payload.get('dataset_endpoint'))}",
        f"Model: {_text(payload.get('model_name'))}",
        f"Chunking: {_text(payload.get('chunking_strategy'))}",
        f"Retrieval top-k: {_text(payload.get('retrieval_top_k'))}",
        f"ABox RAG: {_text(payload.get('use_abox_rag'))}",
        f"ABox top-k: {_text(payload.get('abox_retrieval_top_k'))}",
        f"Reactive ABox discovery: {_text(payload.get('use_reactive_abox_discovery'))}",
        "",
        "QUESTION",
        _text(payload.get("question_asked")),
        "",
        "GENERATION PROMPT",
        _text(payload.get("prompt_generated")),
        "",
        "RETRIEVED ABOX CONTEXT",
        *_render_retrieved_abox_context(payload.get("retrieved_abox_context")),
        "",
        "INITIAL GENERATED QUERY",
        _text(payload.get("llm_generated_query")),
        "",
        "ATTEMPTS",
    ]

    for item in _items(payload.get("correction_iterations")):
        lines.extend(
            [
                "",
                f"Iteration {item.get('iteration')}: {_text(item.get('status'))}",
                f"Validation: {_text(item.get('validation_summary'))}",
                "Query:",
                _text(item.get("query")),
            ]
        )
        errors = item.get("errors")
        if isinstance(errors, list) and errors:
            lines.append("Errors:")
            lines.extend(f"- {_text(error)}" for error in errors)
        execution = item.get("execution")
        if isinstance(execution, dict):
            lines.append(f"Execution: {_text(execution.get('code'))}")
        else:
            lines.append("Execution: not executed")

    lines.extend(
        [
            "",
            "FINAL QUERY",
            _text(payload.get("final_query")),
            "",
            "FINAL RESULTS",
            *_render_execution_result(payload.get("execution_result")),
            "",
            "FINAL ERRORS",
        ]
    )
    errors = payload.get("errors")
    if isinstance(errors, list) and errors:
        lines.extend(f"- {_text(error)}" for error in errors)
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def _render_execution_result(value: object) -> list[str]:
    if not isinstance(value, dict):
        return ["No execution result was recorded."]

    if "boolean" in value:
        return [f"ASK result: {_text(value.get('boolean'))}"]

    head = value.get("head")
    vars_ = head.get("vars") if isinstance(head, dict) else None
    results = value.get("results")
    bindings = results.get("bindings") if isinstance(results, dict) else None
    if not isinstance(vars_, list) or not isinstance(bindings, list):
        return [json.dumps(value, indent=2)]

    clean_vars = [_text(var) for var in vars_]
    lines = [f"Variables: {', '.join(clean_vars)}", f"Rows: {len(bindings)}"]
    for index, row in enumerate(bindings[:MAX_RENDERED_ROWS], 1):
        if not isinstance(row, dict):
            continue
        values = []
        for var in clean_vars:
            cell = row.get(var)
            if isinstance(cell, dict):
                values.append(f"{var}={_text(cell.get('value'))}")
            else:
                values.append(f"{var}=")
        lines.append(f"{index}. " + " | ".join(values))
    if len(bindings) > MAX_RENDERED_ROWS:
        lines.append(f"... {len(bindings) - MAX_RENDERED_ROWS} more rows not shown")
    return lines


def _render_retrieved_abox_context(value: object) -> list[str]:
    if not isinstance(value, list) or not value:
        return ["None"]
    lines: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        display_name = _text(item.get("display_name"))
        uri = _text(item.get("uri"))
        score = _text(item.get("score"))
        lines.append(f"- rank={_text(item.get('rank'))} score={score} name={display_name} uri={uri}")
    return lines or ["None"]


def _items(value: object) -> list[dict[str, object]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _text(value: object) -> str:
    return "" if value is None else str(value)