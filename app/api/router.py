from fastapi import APIRouter

from app.api.routes.health import router as health_router
from app.api.routes.ontology import router as ontology_router
from app.api.routes.query import router as query_router
from app.api.routes.root import router as root_router


api_router = APIRouter()
api_router.include_router(root_router)
api_router.include_router(health_router)
api_router.include_router(ontology_router)
api_router.include_router(query_router)
