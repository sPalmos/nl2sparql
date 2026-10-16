"""Small utility routes for the static UI and active package artifacts."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse

from app.core.config import settings
from app.domain.package import PackageNotFoundError, get_active_package
from app.domain.runtime.query_trace import read_recent_query_traces


router = APIRouter(tags=["root"])
_APP_DIR = Path(__file__).resolve().parents[2]
_STATIC_INDEX = _APP_DIR / "static" / "index.html"


def _active_package_dir() -> Path:
    try:
        return get_active_package(settings.ontology_packages_path)
    except PackageNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/", response_class=FileResponse)
async def read_root() -> FileResponse:
    """Serve the static HTML page used for manual interaction."""
    return FileResponse(_STATIC_INDEX)


@router.get("/load-log", response_class=PlainTextResponse)
async def read_load_log() -> PlainTextResponse:
    path = _active_package_dir() / "logs" / "onboard.log"
    if not path.exists():
        raise HTTPException(status_code=404, detail="load.log not found")
    return PlainTextResponse(path.read_text(encoding="utf-8"))


@router.get("/metadata", response_class=PlainTextResponse)
async def read_metadata() -> PlainTextResponse:
    path = _active_package_dir() / "metadata.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="metadata.json not found")
    return PlainTextResponse(path.read_text(encoding="utf-8"))


QUERY_LOG_RECENT_ENTRIES = 50


@router.get("/query-pipeline-log", response_class=PlainTextResponse)
async def read_query_pipeline_log(limit: int = QUERY_LOG_RECENT_ENTRIES) -> PlainTextResponse:
    """Return the most recent query traces as a JSON array.

    The log is stored as JSON Lines and can grow large, so only the latest
    entries are returned (default 50, adjustable via `?limit=`). The response
    keeps the single JSON array shape the interface has always received.
    """
    path = _active_package_dir() / "logs" / "query.log"
    if not path.exists():
        raise HTTPException(status_code=404, detail="query.log not found")
    entries = read_recent_query_traces(path, limit=max(1, limit))
    return PlainTextResponse(json.dumps(entries, indent=2, ensure_ascii=False))