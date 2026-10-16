"""HTTP route for running the runtime query pipeline against the active package."""

from typing import Any
from typing import Literal

from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, status

from app.core.config import settings
from app.domain.package import DomainError, PackageNotFoundError, get_active_package
from app.domain.runtime import run_query_pipeline


router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    """Request body for the runtime NL-to-SPARQL route."""

    question: str = Field(min_length=1)
    model: str | None = Field(default=None, min_length=1)
    k: int | None = Field(default=None, ge=1)
    abox_rag: bool = False
    abox_k: int | None = Field(default=None, ge=1)
    reactive_abox_discovery: bool = False
    chunking: Literal["class_based", "property_based", "composite"] | None = None
    corrections: int | None = Field(default=None, ge=1)


class QueryResponse(BaseModel):
    """Structured runtime response returned by the query pipeline."""

    question: str
    dataset_name: str
    dataset_endpoint: str
    model_name: str
    retrieved_context: list[dict[str, Any]]
    retrieved_abox_context: list[dict[str, Any]]
    chunking_strategy: str
    retrieval_top_k: int
    abox_retrieval_top_k: int
    use_abox_rag: bool
    use_reactive_abox_discovery: bool
    correction_max_iterations: int
    generated_sparql: str | None
    validated_sparql: str | None
    corrected_sparql: str | None
    execution_result: dict[str, Any] | None
    status: str
    errors: list[str] | None
    trace_path: str
    readable_trace_path: str


@router.post("", response_model=QueryResponse)
async def run_query(request: QueryRequest) -> dict[str, object]:
    """Run the runtime query pipeline for one natural-language question."""
    try:
        result = await run_query_pipeline(
            request.question,
            get_active_package(settings.ontology_packages_path),
            model=request.model,
            k=request.k,
            chunking=request.chunking,
            use_abox_rag=request.abox_rag,
            abox_k=request.abox_k,
            use_reactive_abox_discovery=request.reactive_abox_discovery,
            corrections=request.corrections,
        )
        return result.to_dict()
    except PackageNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except DomainError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
