"""Run the top-level ontology onboarding workflows.

File onboarding loads RDF, prepares the final graph, builds `ontology_context.json`,
writes package artifacts, builds the RAG index, uploads the package data to Fuseki,
and activates the package. Endpoint onboarding follows the same package/index flow
but uses the existing SPARQL endpoint instead of creating a Fuseki dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import json

from app.clients.fuseki import FusekiService
from app.core.config import settings
from app.domain.ontology.graph_preparation import prepare_final_graph
from app.domain.ontology.ontology_context import build_ontology_context
from app.domain.ontology.package_activation import build_fuseki_uploads_from_package
from app.domain.ontology.package_writer import (
    OntologyPackageArtifacts,
    append_onboard_log,
    write_ontology_package,
)
from app.domain.ontology.source_loader import SUPPORTED_SUFFIXES, load_ontology_file, load_sparql_endpoint
from app.domain.package import get_active_package, set_active_package
from app.domain.rag import build_abox_index, build_all_indexes


@dataclass(frozen=True)
class OnboardingResult:
    """Structured result returned by the shared onboarding flow."""

    package_dir: Path
    dataset_name: str | None
    dataset_endpoint: str | None
    query_endpoint: str | None
    source_mode: str
    chunks_path: Path
    index_path: Path
    chunk_count: int
    abox_chunks_path: Path | None = None
    abox_index_path: Path | None = None
    abox_chunk_count: int = 0


async def onboard_ontology_file(
    source_path: str | Path,
    *,
    packages_root: str | Path,
    fuseki_service: FusekiService,
    source_filename: str | None = None,
    package_name: str | None = None,
    activate_package: bool = True,
    build_abox: bool | None = None,
    status_callback: callable | None = None,
) -> OnboardingResult:
    """Build one package from a file, index it, upload it to Fuseki, and optionally activate it."""
    source_file = Path(source_path).resolve()
    suffix = source_file.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError("Only .ttl, .owl, and .rdf files are supported")

    effective_source_filename = source_filename or source_file.name
    ontology_name = _slugify_name(package_name) if package_name else _slugify_filename(effective_source_filename)
    packages_root_path = Path(packages_root).resolve()
    dataset_name = _unique_timestamped_name(packages_root_path, ontology_name)
    package_dir = packages_root_path / dataset_name
    query_endpoint = f"{fuseki_service.dataset_endpoint(dataset_name)}/query"

    _emit_status(status_callback, package_dir, "loading_source", source=str(source_file))
    source = await load_ontology_file(
        str(source_file),
        source_filename=effective_source_filename,
    )
    _emit_status(status_callback, package_dir, "preparing_graph", source=str(source_file))
    final_graph = await prepare_final_graph(source.graph)
    ontology_context = build_ontology_context(
        final_graph.graph,
        ontology_name=ontology_name,
        source_filename=effective_source_filename,
    )
    _emit_status(status_callback, package_dir, "writing_package", source=str(source_file))
    artifacts = write_ontology_package(
        package_dir=package_dir,
        source=source,
        final_graph=final_graph,
        ontology_name=ontology_name,
        ontology_context=ontology_context,
        dataset_name=dataset_name,
        query_endpoint=query_endpoint,
    )

    _emit_status(status_callback, package_dir, "building_indexes")
    artifact_results = build_all_indexes(package_dir)
    default_artifact_result = _default_index_result(artifact_results, settings.default_chunking_strategy)
    abox_artifact_result = None
    should_build_abox = build_abox if build_abox is not None else settings.default_build_abox_index
    if should_build_abox:
        _emit_status(status_callback, package_dir, "building_abox_index")
        abox_artifact_result = build_abox_index(package_dir, final_graph.graph)

    _emit_status(status_callback, package_dir, "uploading_to_fuseki", dataset_name=dataset_name)
    uploads = build_fuseki_uploads_from_package(package_dir, dataset_name=dataset_name)
    await fuseki_service.replace_dataset(
        dataset_name=dataset_name,
        files=uploads,
        previous_dataset_name=_previous_dataset_name(packages_root),
    )

    if activate_package:
        set_active_package(packages_root, package_dir)
        _emit_status(status_callback, package_dir, "package_activated", activated_package=str(package_dir))

    _emit_status(
        status_callback,
        package_dir,
        "onboarding_completed",
        completed_package=str(package_dir),
        dataset_name=dataset_name,
        chunk_count=default_artifact_result.chunk_count,
    )
    return OnboardingResult(
        package_dir=package_dir,
        dataset_name=dataset_name,
        dataset_endpoint=fuseki_service.dataset_endpoint(dataset_name),
        query_endpoint=query_endpoint,
        source_mode="file",
        chunks_path=default_artifact_result.chunks_path,
        index_path=default_artifact_result.index_path,
        chunk_count=default_artifact_result.chunk_count,
        abox_chunks_path=getattr(abox_artifact_result, "chunks_path", None),
        abox_index_path=getattr(abox_artifact_result, "index_path", None),
        abox_chunk_count=getattr(abox_artifact_result, "chunk_count", 0),
    )


async def onboard_sparql_endpoint(
    endpoint: str,
    *,
    packages_root: str | Path,
    package_name: str | None = None,
    activate_package: bool = True,
    build_abox: bool | None = None,
    status_callback: callable | None = None,
) -> OnboardingResult:
    """Build one package from an existing SPARQL endpoint and optionally activate it."""
    packages_root_path = Path(packages_root).resolve()
    ontology_name = _slugify_name(package_name) if package_name else _slugify_endpoint(endpoint)
    package_dir = packages_root_path / _unique_timestamped_name(packages_root_path, ontology_name)

    _emit_status(status_callback, package_dir, "loading_source", source=endpoint)
    source = await load_sparql_endpoint(endpoint)
    _emit_status(status_callback, package_dir, "preparing_graph", source=endpoint)
    final_graph = await prepare_final_graph(source.graph, resolve_missing_schemas=False)
    ontology_context = build_ontology_context(
        final_graph.graph,
        ontology_name=ontology_name,
        source_filename=endpoint,
    )
    _emit_status(status_callback, package_dir, "writing_package", source=endpoint)
    write_ontology_package(
        package_dir=package_dir,
        source=source,
        final_graph=final_graph,
        ontology_name=ontology_name,
        ontology_context=ontology_context,
        dataset_name=None,
        query_endpoint=endpoint,
    )

    _emit_status(status_callback, package_dir, "building_indexes")
    artifact_results = build_all_indexes(package_dir)
    default_artifact_result = _default_index_result(artifact_results, settings.default_chunking_strategy)
    abox_artifact_result = None
    should_build_abox = build_abox if build_abox is not None else settings.default_build_abox_index
    if should_build_abox:
        _emit_status(status_callback, package_dir, "building_abox_index")
        abox_artifact_result = build_abox_index(package_dir, final_graph.graph)

    if activate_package:
        set_active_package(packages_root, package_dir)
        _emit_status(status_callback, package_dir, "package_activated", activated_package=str(package_dir))

    _emit_status(
        status_callback,
        package_dir,
        "onboarding_completed",
        completed_package=str(package_dir),
        dataset_name=None,
        chunk_count=default_artifact_result.chunk_count,
    )
    return OnboardingResult(
        package_dir=package_dir,
        dataset_name=None,
        dataset_endpoint=None,
        query_endpoint=endpoint,
        source_mode="sparql_endpoint",
        chunks_path=default_artifact_result.chunks_path,
        index_path=default_artifact_result.index_path,
        chunk_count=default_artifact_result.chunk_count,
        abox_chunks_path=getattr(abox_artifact_result, "chunks_path", None),
        abox_index_path=getattr(abox_artifact_result, "index_path", None),
        abox_chunk_count=getattr(abox_artifact_result, "chunk_count", 0),
    )


def _previous_dataset_name(packages_root: str | Path) -> str | None:
    try:
        active_root = get_active_package(packages_root)
    except Exception:
        return None

    metadata_file = active_root / "metadata.json"
    if not metadata_file.exists():
        return None

    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    dataset_name = metadata.get("dataset_name")
    return dataset_name if isinstance(dataset_name, str) and dataset_name.strip() else None


def _default_index_result(results: list[object], chunking: str):
    for result in results:
        if getattr(result, "chunking", None) == chunking:
            return result
    return results[0]


def _slugify_filename(filename: str) -> str:
    stem = Path(filename).stem.lower()
    return _slugify_name(stem)


def _slugify_endpoint(endpoint: str) -> str:
    text = endpoint.rstrip("/")
    tail = text.rsplit("/", 1)[-1]
    return _slugify_name(tail) or "endpoint"


def _slugify_name(value: str | None) -> str:
    if not value:
        return "ontology"
    normalized = value.lower().replace("_", "-")
    allowed = "".join(char if char.isalnum() or char == "-" else "-" for char in normalized)
    return "-".join(part for part in allowed.split("-") if part) or "ontology"


def _unique_timestamped_name(root: Path, base_name: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M")
    stem = f"{base_name}-{timestamp}"
    candidate = stem
    index = 2
    while (root / candidate).exists():
        candidate = f"{stem}-{index}"
        index += 1
    return candidate


def _emit_status(
    callback: callable | None,
    package_dir: Path,
    event: str,
    **details: object,
) -> None:
    _append_onboard_log(package_dir, event, **details)
    if callback is not None:
        callback(event, **details)


def _append_onboard_log(package_dir: Path, event: str, **details: object) -> None:
    append_onboard_log(package_dir / "logs" / "onboard.log", event, **details)
