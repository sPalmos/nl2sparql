"""Write ontology package artifacts to disk.

This module owns package filesystem side effects: ensuring directories exist,
copying the source ontology file, writing resolved schema files, and persisting
`metadata.json`, `ontology_context.json`, `settings.json`, and onboarding log
entries. It does not parse RDF, prepare graphs, build chunks, or upload to
Fuseki.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import json

from app.domain.package import (
    ensure_package_dirs,
    logs_dir,
    metadata_path,
    ontology_context_path,
    schemas_dir,
    settings_path,
    source_path,
    write_json_file,
)
from app.domain.ontology.graph_preparation import FinalGraph, ResolvedSchemaFile
from app.domain.ontology.source_loader import LoadedOntologySource


@dataclass(frozen=True)
class OntologyPackageArtifacts:
    """Files and payloads written before RAG indexing."""

    package_dir: Path
    metadata: dict[str, object]
    ontology_context: dict[str, object]
    settings: dict[str, object]
    source_path: Path | None
    resolved_schemas: list[ResolvedSchemaFile]


def write_ontology_package(
    *,
    package_dir: str | Path,
    source: LoadedOntologySource,
    final_graph: FinalGraph,
    ontology_name: str,
    ontology_context: dict[str, object],
    dataset_name: str | None,
    query_endpoint: str | None,
) -> OntologyPackageArtifacts:
    """Persist source, metadata, settings, and ontology context artifacts."""
    root = ensure_package_dirs(package_dir)
    copied_source_path = _copy_source_file(root, source)
    _write_resolved_schemas(root, final_graph.resolved_schemas.resolved_files)

    timestamp = datetime.now(UTC)
    metadata = _build_metadata(
        attempted_at=timestamp,
        ontology_name=ontology_name,
        source_filename=source.source_name,
        ontology_file=copied_source_path.relative_to(root).as_posix() if copied_source_path else None,
        dataset_name=dataset_name,
        query_endpoint=query_endpoint,
        final_graph=final_graph,
        source_mode=source.source_mode,
    )
    write_json_file(metadata_path(root), metadata)
    write_json_file(ontology_context_path(root), ontology_context)

    settings_payload = _settings_payload(
        source_mode=source.source_mode,
        query_endpoint=query_endpoint,
        dataset_name=dataset_name,
    )
    write_json_file(settings_path(root), settings_payload)
    append_onboard_log(logs_dir(root) / "onboard.log", "metadata_extracted", metadata=metadata)

    return OntologyPackageArtifacts(
        package_dir=root,
        metadata=metadata,
        ontology_context=ontology_context,
        settings=settings_payload,
        source_path=copied_source_path,
        resolved_schemas=final_graph.resolved_schemas.resolved_files,
    )


def append_onboard_log(path: Path, event: str, **details: object) -> None:
    """Append one JSON event to the onboarding log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": _minute_timestamp(datetime.now(UTC)),
        "event": event,
        **details,
    }
    with path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(entry))
        log_file.write("\n")


def _copy_source_file(package_dir: Path, source: LoadedOntologySource) -> Path | None:
    if source.source_mode != "file":
        return None
    if source.content is None or source.suffix is None:
        raise ValueError("File source is missing content or suffix")
    copied_source_path = source_path(package_dir, source.suffix)
    copied_source_path.write_bytes(source.content)
    return copied_source_path


def _write_resolved_schemas(package_dir: Path, schema_files: list[ResolvedSchemaFile]) -> None:
    for schema_file in schema_files:
        (schemas_dir(package_dir) / schema_file.filename).write_bytes(schema_file.content)


def _build_metadata(
    *,
    attempted_at: datetime,
    ontology_name: str,
    source_filename: str,
    ontology_file: str | None,
    dataset_name: str | None,
    query_endpoint: str | None,
    final_graph: FinalGraph,
    source_mode: str,
) -> dict[str, object]:
    files_loaded = []
    if ontology_file:
        files_loaded.append(ontology_file)
    files_loaded.extend(
        f"ontology/schemas/{schema_file.filename}"
        for schema_file in final_graph.resolved_schemas.resolved_files
    )

    return {
        "loaded_at": _minute_timestamp(attempted_at),
        "ontology_name": ontology_name,
        "source_filename": source_filename,
        "source_mode": source_mode,
        "dataset_name": dataset_name,
        "query_endpoint": query_endpoint,
        "ontology_file": ontology_file,
        "mode": final_graph.mode,
        "initial_graph": {
            "triple_count": len(final_graph.initial_graph),
            "classes_count": final_graph.detection.classes_count,
            "instances_count": final_graph.detection.instances_count,
            "properties_count": final_graph.detection.properties_count,
        },
        "schema_coverage": {
            "status": final_graph.coverage.status,
            "missing_class_uris": final_graph.coverage.missing_class_uris,
            "missing_namespaces": final_graph.coverage.missing_namespaces,
        },
        "files_loaded": files_loaded,
        "resolved_schemas": [
            {
                "source_namespace": schema_file.source_namespace,
                "url": schema_file.url,
                "local_file": f"ontology/schemas/{schema_file.filename}",
            }
            for schema_file in final_graph.resolved_schemas.resolved_files
        ],
        "triple_count": len(final_graph.graph),
    }


def _settings_payload(
    *,
    source_mode: str,
    query_endpoint: str | None,
    dataset_name: str | None,
) -> dict[str, object]:
    return {
        "source_mode": source_mode,
        "query_endpoint": query_endpoint,
        "dataset_name": dataset_name,
    }


def _minute_timestamp(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%MZ")
