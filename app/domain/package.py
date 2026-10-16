"""Small path and JSON helpers for ontology packages."""

from __future__ import annotations

from pathlib import Path
import json


ACTIVE_PACKAGE_FILENAME = ".active_package"


class DomainError(Exception):
    """Base exception for reusable domain logic."""


class PackageNotFoundError(DomainError):
    """Raised when a requested ontology package or artifact is missing."""


class InvalidPackageError(DomainError):
    """Raised when an ontology package has an invalid format."""


def resolve_package_dir(package_dir: str | Path) -> Path:
    """Return the absolute ontology package path."""
    return Path(package_dir).resolve()


def ensure_package_dirs(package_dir: str | Path) -> Path:
    """Create the standard ontology-package directories and return the root path."""
    root = resolve_package_dir(package_dir)
    root.mkdir(parents=True, exist_ok=True)
    ontology_dir(root).mkdir(parents=True, exist_ok=True)
    schemas_dir(root).mkdir(parents=True, exist_ok=True)
    logs_dir(root).mkdir(parents=True, exist_ok=True)
    indexes_dir(root).mkdir(parents=True, exist_ok=True)
    return root


def ontology_dir(package_dir: str | Path) -> Path:
    return resolve_package_dir(package_dir) / "ontology"


def schemas_dir(package_dir: str | Path) -> Path:
    return ontology_dir(package_dir) / "schemas"


def logs_dir(package_dir: str | Path) -> Path:
    return resolve_package_dir(package_dir) / "logs"


def indexes_dir(package_dir: str | Path) -> Path:
    return resolve_package_dir(package_dir) / "indexes"


def index_strategy_dir(package_dir: str | Path, chunking: str) -> Path:
    return indexes_dir(package_dir) / chunking


def metadata_path(package_dir: str | Path) -> Path:
    return resolve_package_dir(package_dir) / "metadata.json"


def ontology_context_path(package_dir: str | Path) -> Path:
    return resolve_package_dir(package_dir) / "ontology_context.json"


def settings_path(package_dir: str | Path) -> Path:
    return resolve_package_dir(package_dir) / "settings.json"


def onboard_log_path(package_dir: str | Path) -> Path:
    return logs_dir(package_dir) / "onboard.log"


def query_log_path(package_dir: str | Path) -> Path:
    return logs_dir(package_dir) / "query.log"


def query_readable_latest_path(package_dir: str | Path) -> Path:
    return logs_dir(package_dir) / "query-latest.txt"


def query_readable_runs_dir(package_dir: str | Path) -> Path:
    return logs_dir(package_dir) / "query-runs"


def source_path(package_dir: str | Path, suffix: str) -> Path:
    return ontology_dir(package_dir) / f"source{suffix}"


def chunks_path(package_dir: str | Path, chunking: str = "class_based") -> Path:
    return index_strategy_dir(package_dir, chunking) / "chunks.json"


def index_path(package_dir: str | Path, chunking: str = "class_based") -> Path:
    return index_strategy_dir(package_dir, chunking) / "index.faiss"


def abox_index_dir(package_dir: str | Path) -> Path:
    return indexes_dir(package_dir) / "abox"


def abox_chunks_path(package_dir: str | Path) -> Path:
    return abox_index_dir(package_dir) / "chunks.json"


def abox_index_path(package_dir: str | Path) -> Path:
    return abox_index_dir(package_dir) / "index.faiss"


def read_json_file(path: Path) -> dict[str, object]:
    """Read a JSON object from disk."""
    if not path.exists():
        raise PackageNotFoundError(f"Required artifact not found: {path.as_posix()}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InvalidPackageError(f"Failed to load JSON artifact: {path.as_posix()}") from exc

    if not isinstance(payload, dict):
        raise InvalidPackageError(f"Expected a JSON object in: {path.as_posix()}")
    return payload


def write_json_file(path: Path, payload: dict[str, object]) -> None:
    """Write one JSON object to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json_list(path: Path) -> list[dict[str, object]]:
    """Read a JSON list of objects from disk."""
    if not path.exists():
        raise PackageNotFoundError(f"Required artifact not found: {path.as_posix()}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InvalidPackageError(f"Failed to load JSON artifact: {path.as_posix()}") from exc

    if not isinstance(payload, list):
        raise InvalidPackageError(f"Expected a JSON list in: {path.as_posix()}")
    return [item for item in payload if isinstance(item, dict)]


def active_package_pointer(packages_root: str | Path) -> Path:
    return resolve_package_dir(packages_root) / ACTIVE_PACKAGE_FILENAME


def set_active_package(packages_root: str | Path, package_dir: str | Path) -> None:
    pointer = active_package_pointer(packages_root)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(str(resolve_package_dir(package_dir)), encoding="utf-8")


def get_active_package(packages_root: str | Path) -> Path:
    pointer = active_package_pointer(packages_root)
    if not pointer.exists():
        raise PackageNotFoundError("No active ontology package is set")

    target = Path(pointer.read_text(encoding="utf-8").strip()).resolve()
    if not target.exists():
        raise PackageNotFoundError(f"Active ontology package not found: {target.as_posix()}")
    return target
