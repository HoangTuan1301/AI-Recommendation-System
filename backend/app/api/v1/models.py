"""Models Registry Endpoint."""

from fastapi import APIRouter
from backend.app.schemas.model import ModelsListResponse
from backend.app.services.model_service import model_service

router = APIRouter(prefix="/models", tags=["Models"])


@router.get("", response_model=ModelsListResponse)
def get_models():
    """Retrieve metadata and score semantics for all 10 AI recommendation models."""
    models = model_service.list_models()
    return ModelsListResponse(total=len(models), models=models)
