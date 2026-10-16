"""Activate existing ontology packages for runtime querying.

Activation is the explicit operation that makes an ontology package runnable by
the CLI and API. Local file packages are reloaded into the managed Fuseki
instance from package artifacts. SPARQL endpoint packages are external pointers,
so activation only marks the package active.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.clients.fuseki import FusekiService, FusekiUploadPayload
from app.domain.ontology.package_writer import append_onboard_log
from app.domain.package import (
    InvalidPackageError,
    PackageNotFoundError,
    get_active_package,
    metadata_path,
    read_json_file,
    resolve_package_dir,
    schemas_dir,
    set_active_package,
    settings_path,
)


@dataclass(frozen=True)
class ActivationResult:
    """Result of activating one ontology package."""

    package_dir: Path
    source_mode: str
    dataset_name: str | None
    query_endpoint: str | None
    reloaded: bool


async def activate_package(
    package: str | Path,
    *,
    packages_root: str | Path,
    fuseki_service: FusekiService,
) -> ActivationResult:
    """Activate a package and reload local package data into Fuseki when needed."""
    package_dir = resolve_package_reference(package, packages_root)
    metadata = read_json_file(metadata_path(package_dir))
    settings_payload = read_json_file(settings_path(package_dir))

    source_mode = _required_string(metadata, settings_payload, key="source_mode")
    query_endpoint = _optional_string(settings_payload, metadata, key="query_endpoint")

    previous_dataset_name = _active_dataset_name(packages_root)

    if source_mode == "sparql_endpoint":
        if previous_dataset_name:
            await fuseki_service.delete_dataset(previous_dataset_name, ignore_missing=True)
        set_active_package(packages_root, package_dir)
        _append_activation_log(package_dir, reloaded=False, query_endpoint=query_endpoint)
        return ActivationResult(
            package_dir=package_dir,
            source_mode=source_mode,
            dataset_name=None,
            query_endpoint=query_endpoint,
            reloaded=False,
        )

    if source_mode != "file":
        raise InvalidPackageError(f"Unsupported package source_mode: {source_mode}")

    dataset_name = _required_string(metadata, settings_payload, key="dataset_name")
    uploads = build_fuseki_uploads_from_package(package_dir, dataset_name=dataset_name)
    await fuseki_service.reload_active_dataset(
        dataset_name=dataset_name,
        files=uploads,
        previous_dataset_name=previous_dataset_name,
    )
    set_active_package(packages_root, package_dir)
    _append_activation_log(
        package_dir,
        reloaded=True,
        dataset_name=dataset_name,
        query_endpoint=query_endpoint,
        uploaded_files=[payload.filename for payload in uploads],
    )
    return ActivationResult(
        package_dir=package_dir,
        source_mode=source_mode,
        dataset_name=dataset_name,
        query_endpoint=query_endpoint,
        reloaded=True,
    )


def resolve_package_reference(package: str | Path, packages_root: str | Path) -> Path:
    """Resolve either an explicit path or a package directory name under packages_root."""
    raw = Path(package)
    candidates = []
    if raw.is_absolute() or raw.exists():
        candidates.append(raw)
    candidates.append(Path(packages_root) / raw)

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists() and resolved.is_dir():
            return resolved

    raise PackageNotFoundError(f"Ontology package not found: {package}")


def build_fuseki_uploads_from_package(
    package_dir: str | Path,
    *,
    dataset_name: str,
) -> list[FusekiUploadPayload]:
    """Build Fuseki upload payloads from a persisted local package."""
    root = resolve_package_dir(package_dir)
    metadata = read_json_file(metadata_path(root))

    ontology_file = metadata.get("ontology_file")
    source_file = root / ontology_file if isinstance(ontology_file, str) and ontology_file.strip() else None
    if source_file is None or not source_file.exists():
        source_candidates = sorted((root / "ontology").glob("source.*"))
        if not source_candidates:
            raise PackageNotFoundError(f"No copied ontology source found in package: {root.as_posix()}")
        source_file = source_candidates[0]

    source_filename = metadata.get("source_filename")
    upload_filename = source_filename if isinstance(source_filename, str) and source_filename.strip() else source_file.name
    uploads = [
        FusekiUploadPayload(
            dataset_name=dataset_name,
            filename=upload_filename,
            content=source_file.read_bytes(),
        )
    ]

    schema_files = _schema_files(root, metadata)
    uploads.extend(
        FusekiUploadPayload(
            dataset_name=dataset_name,
            filename=schema_file.name,
            content=schema_file.read_bytes(),
        )
        for schema_file in schema_files
    )
    return uploads


def _schema_files(root: Path, metadata: dict[str, object]) -> list[Path]:
    declared = metadata.get("resolved_schemas")
    files: list[Path] = []
    if isinstance(declared, list):
        for item in declared:
            if not isinstance(item, dict):
                continue
            local_file = item.get("local_file")
            if isinstance(local_file, str) and local_file.strip():
                path = root / local_file
                if path.exists() and path.is_file():
                    files.append(path)
    if files:
        return files
    schema_root = schemas_dir(root)
    if not schema_root.exists():
        return []
    return sorted(path for path in schema_root.iterdir() if path.is_file())


def _active_dataset_name(packages_root: str | Path) -> str | None:
    try:
        active_root = get_active_package(packages_root)
        metadata = read_json_file(metadata_path(active_root))
    except Exception:
        return None
    dataset_name = metadata.get("dataset_name")
    return dataset_name if isinstance(dataset_name, str) and dataset_name.strip() else None


def _required_string(*payloads: dict[str, object], key: str) -> str:
    value = _optional_string(*payloads, key=key)
    if value is None:
        raise InvalidPackageError(f"Package is missing required value: {key}")
    return value


def _optional_string(*payloads: dict[str, object], key: str) -> str | None:
    for payload in payloads:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _append_activation_log(package_dir: Path, **details: object) -> None:
    append_onboard_log(package_dir / "logs" / "activation.log", "package_activated", **details)
