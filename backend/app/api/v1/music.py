"""MusicAI REST API Router.

Exposes endpoints for song-level recommendation, catalog inspection,
listening history, model registry status, and health diagnostics.
"""

import logging
from typing import Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Query, status

from backend.app.schemas.music import (
    MusicHealthResponse,
    MusicModelsListResponse,
    MusicRecommendationResponse,
    SongDetailResponse,
    UserHistoryResponse,
)
from backend.app.services.music_service import music_service
from src.music.models.registry import ModelNotFoundError, CheckpointError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["MusicAI - Music Recommendation"])


class MusicRecommendationRequest(BaseModel):
    """Request body for POST recommendation endpoints."""
    user_id: str = Field(..., min_length=1, description="Target user identifier")
    model: str = Field(default="item_cf", description="Model slug identifier")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of songs to recommend")


@router.get(
    "/health",
    response_model=MusicHealthResponse,
    summary="MusicAI Health & Diagnostics",
    description="Returns domain statistics, catalog size, model availability, and service readiness.",
)
def get_music_health():
    """Return health and status for MusicAI."""
    try:
        return music_service.get_health()
    except Exception as exc:
        logger.error(f"Error in music health endpoint: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(exc)}",
        )


@router.get(
    "/models",
    response_model=MusicModelsListResponse,
    summary="List MusicAI Models",
    description="Retrieves metadata, family, description, and status for all 6 song recommendation models.",
)
def get_music_models():
    """Retrieve all 6 MusicAI models."""
    try:
        return music_service.list_models()
    except Exception as exc:
        logger.error(f"Error listing music models: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not retrieve music models: {str(exc)}",
        )


@router.get(
    "/recommendations",
    response_model=MusicRecommendationResponse,
    summary="Generate Song Recommendations",
    description="Generates Top-K song recommendations for a user with the requested AI model via GET query params.",
)
def get_music_recommendations(
    user_id: str = Query(
        ...,
        description="Target user ID from Million Song Dataset (or any string for cold-start demo)",
        min_length=1,
    ),
    model: str = Query(
        default="item_cf",
        description="Model slug: 'item_cf' (default), 'user_cf', 'ials', 'bpr_mf', 'implicit_neural_cf', 'popularity'",
    ),
    top_k: int = Query(
        default=10,
        ge=1,
        le=100,
        description="Number of songs to recommend (between 1 and 100)",
    ),
):
    """Generate song recommendations for a user via GET."""
    try:
        return music_service.get_recommendations(
            user_id=user_id,
            model_name=model,
            top_k=top_k,
        )
    except ModelNotFoundError as mnf_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(mnf_err),
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except CheckpointError as ckpt_err:
        logger.error(f"Checkpoint failure during recommendation: {ckpt_err}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Recommendation engine checkpoint error: {str(ckpt_err)}",
        )
    except Exception as exc:
        logger.error(f"Unexpected error in recommendation endpoint: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recommendation inference failed: {str(exc)}",
        )


@router.post(
    "/recommendations",
    response_model=MusicRecommendationResponse,
    summary="Generate Song Recommendations (POST)",
    description="Generates Top-K song recommendations for a user with the requested AI model via JSON body.",
)
@router.post(
    "/recommend",
    response_model=MusicRecommendationResponse,
    summary="Generate Song Recommendations (POST alias)",
    description="Legacy alias matching /api/music/recommend for backwards compatibility.",
)
def post_music_recommendations(req: MusicRecommendationRequest):
    """Generate song recommendations for a user via POST."""
    try:
        return music_service.get_recommendations(
            user_id=req.user_id,
            model_name=req.model,
            top_k=req.top_k,
        )
    except ModelNotFoundError as mnf_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(mnf_err),
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except CheckpointError as ckpt_err:
        logger.error(f"Checkpoint failure during recommendation: {ckpt_err}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Recommendation engine checkpoint error: {str(ckpt_err)}",
        )
    except Exception as exc:
        logger.error(f"Unexpected error in recommendation endpoint: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recommendation inference failed: {str(exc)}",
        )


@router.get(
    "/songs/{song_id}",
    response_model=SongDetailResponse,
    summary="Get Song Catalog Detail",
    description="Retrieves metadata and training split listening statistics for a specific song ID.",
)
def get_song_detail(song_id: str):
    """Retrieve details for a single song."""
    clean_id = song_id.strip()
    song = music_service.get_song(clean_id)
    if not song:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Song with ID '{clean_id}' not found in catalog.",
        )
    return song


@router.get(
    "/users/{user_id}/history",
    response_model=UserHistoryResponse,
    summary="Get User Training History",
    description="Retrieves listening history strictly from training split interactions for demonstration.",
)
@router.get(
    "/users/sample_users/{user_id}/history",
    response_model=UserHistoryResponse,
    summary="Get User Training History (Alias)",
    description="Alias route matching /users/sample_users/{user_id}/history for client compatibility.",
)
def get_user_history(
    user_id: str,
    limit: int = Query(
        default=50,
        ge=1,
        le=500,
        description="Maximum history tracks to return",
    ),
):
    """Retrieve training listening history for a user."""
    clean_uid = user_id.strip()
    history = music_service.get_user_history(clean_uid, limit=limit)
    if not history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID '{clean_uid}' has no recorded listening history.",
        )
    return history
