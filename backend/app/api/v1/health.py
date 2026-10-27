"""Health and Diagnostics Endpoint."""

from fastapi import APIRouter
from backend.app.services.model_service import model_service
from backend.app.services.movie_service import movie_service

router = APIRouter(tags=["Health"])


@router.get("/health")
def get_health():
    """Return system status and readiness."""
    return {
        "status": "healthy",
        "service": "CineAI Recommendation Backend",
        "dataset": "MovieLens 100K",
        "catalog_movies": len(movie_service.movies_by_id),
        "available_models": len(model_service.registry),
        "cached_models": list(model_service.cached_models.keys()),
    }
