"""Execute SPARQL queries and format execution trace stages.

This module is the only runtime module that talks to the SPARQL endpoint. It
posts a query to the configured endpoint and returns the endpoint JSON response.
It also formats execution success/failure as the same stage-result shape used by
formal validation so the pipeline trace can record execution uniformly.
"""

from __future__ import annotations

import httpx

from app.core.config import settings
from app.domain.runtime.validation import ValidationStageResult


async def preflight_sparql_endpoint(endpoint: str) -> None:
    """Verify the configured SPARQL endpoint is reachable before a query run."""
    if not endpoint:
        raise ValueError("No SPARQL query endpoint is configured")

    async with httpx.AsyncClient(timeout=settings.fuseki_upload_timeout_seconds) as client:
        response = await client.post(
            endpoint,
            data={"query": "ASK WHERE { ?s ?p ?o }"},
            headers={"Accept": "application/sparql-results+json, application/json"},
        )
        response.raise_for_status()


class SparqlExecutionTimeout(Exception):
    """Raised when a query exceeds the configured per-query execution budget."""


async def execute_sparql_query(
    endpoint: str,
    query: str,
    *,
    timeout_seconds: float | None = None,
) -> dict[str, object]:
    """Execute one SPARQL query against an explicit endpoint URL.

    The endpoint must be the query endpoint, not just the dataset base URL. HTTP
    status failures are propagated via `httpx` exceptions so the pipeline can
    record them as `EXECUTION_ERROR` and decide whether to request correction.

    Execution is bounded by a per-query budget. Generated queries occasionally
    carry pathological cost on large instance graphs, and an unbounded request
    would stall the whole run on a single question. The budget is applied twice:
    as a server-side cap Fuseki honours by abandoning the query, and as a client
    read timeout in case the server does not enforce it. Exceeding it raises
    `SparqlExecutionTimeout`, which the caller records as `EXECUTION_ERROR` in
    the same way as any other endpoint failure, so the correction loop can
    attempt a different query.
    """
    if not endpoint:
        raise ValueError("No SPARQL query endpoint is configured")

    budget = timeout_seconds or settings.fuseki_query_timeout_seconds
    # Fuseki reads its own limit in milliseconds; allow the client a small
    # margin so the server-side abort is the one that normally fires.
    payload = {"query": query, "timeout": str(int(budget * 1000))}

    try:
        async with httpx.AsyncClient(timeout=budget + 10.0) as client:
            response = await client.post(
                endpoint,
                data=payload,
                headers={"Accept": "application/sparql-results+json, application/json"},
            )
            response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise SparqlExecutionTimeout(
            f"Query exceeded the {budget:.0f}s execution budget"
        ) from exc
    return response.json()


def execution_stage_result(error: Exception | None = None) -> ValidationStageResult:
    """Return the trace-stage result for endpoint execution.

    Execution is not part of formal SPARQL validation, but the trace stores it
    beside validation stages. This helper keeps the execution stage codes stable:
    `EXECUTION_OK` for success and `EXECUTION_ERROR` for endpoint failures.
    """
    if error is None:
        return ValidationStageResult(stage="execution", passed=True, code="EXECUTION_OK")
    return ValidationStageResult(
        stage="execution",
        passed=False,
        code="EXECUTION_ERROR",
        message=f"SPARQL endpoint execution failed: {error}",
    )