"""FastAPI application entry point for the ontology-package workflow."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router
from app.core.config import settings


def create_app() -> FastAPI:
    """Create the web app and register the static UI plus API routes."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
    )
    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    app.include_router(api_router)
    return app


app = create_app()
