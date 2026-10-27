"""AI Recommendation Inference Endpoint."""

from fastapi import APIRouter, HTTPException
from backend.app.schemas.recommendation import RecommendationRequest, RecommendationResponse
from backend.app.services.model_service import model_service

router = APIRouter(prefix="/recommendations", tags=["Recommendations"])


@router.post("", response_model=RecommendationResponse)
def create_recommendations(req: RecommendationRequest):
    """Generate Top-N movie recommendations using real AI model inference."""
    try:
        return model_service.get_recommendations(
            user_id=req.user_id,
            model_id=req.model_id,
            top_n=req.top_n,
        )
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(exc)}")
