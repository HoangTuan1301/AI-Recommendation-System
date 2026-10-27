"""Users and Profiles Endpoints."""

from fastapi import APIRouter, HTTPException, Query
from backend.app.schemas.user import UserProfileResponse, UsersListResponse
from backend.app.services.user_service import user_service

router = APIRouter(prefix="/users", tags=["Users"])


@router.get("", response_model=UsersListResponse)
def list_users(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=50, ge=1, le=200),
):
    """Retrieve paginated list of valid MovieLens users with rating summaries."""
    return user_service.list_users(page=page, limit=limit)


@router.get("/{user_id}/profile", response_model=UserProfileResponse)
def get_user_profile(user_id: int):
    """Retrieve user taste profile, favorite genres, and top-rated movies strictly from Train."""
    profile = user_service.get_user_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"User with ID {user_id} not found in training dataset.")
    return profile
