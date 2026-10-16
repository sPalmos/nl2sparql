"""HTTP route for onboarding an uploaded ontology into an ontology package."""

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.clients.fuseki import FusekiService
from app.core.config import settings
from app.domain.ontology import onboard_ontology_file


router = APIRouter(prefix="/ontology", tags=["ontology"])
_FUSEKI_SERVICE = FusekiService()
_SUPPORTED_SUFFIXES = {".ttl", ".owl", ".rdf"}


@router.post("/load")
async def load_ontology(
    file: UploadFile = File(...),
) -> dict[str, str]:
    """Create a package, build all indexes, upload it to Fuseki, and activate it."""
    source_filename = file.filename or "ontology"
    suffix = source_filename[source_filename.rfind("."):].lower() if "." in source_filename else ""
    if suffix not in _SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .ttl, .owl, and .rdf files are supported",
        )

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    from pathlib import Path

    packages_root = Path(settings.ontology_packages_path).resolve()
    package_dir = packages_root / "_incoming"
    package_dir.mkdir(parents=True, exist_ok=True)
    incoming_path = package_dir / f".incoming{suffix}"
    incoming_path.write_bytes(content)

    try:
        result = await onboard_ontology_file(
            incoming_path,
            packages_root=packages_root,
            fuseki_service=_FUSEKI_SERVICE,
            source_filename=source_filename,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    finally:
        if incoming_path.exists():
            incoming_path.unlink()

    return {
        "package_dir": str(result.package_dir),
        "dataset_name": result.dataset_name,
        "endpoint": result.dataset_endpoint,
    }
