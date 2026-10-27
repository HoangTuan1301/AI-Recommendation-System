"""Movies Catalog Endpoints."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from backend.app.schemas.movie import MovieDetail, MoviePagination
from backend.app.services.movie_service import movie_service

router = APIRouter(prefix="/movies", tags=["Movies"])


@router.get("", response_model=MoviePagination)
def list_movies(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
    q: Optional[str] = Query(default=None, description="Search keyword in title"),
    genre: Optional[str] = Query(default=None, description="Filter by genre"),
    sort_by: str = Query(default="popular", description="Sort by 'popular', 'rating', 'title', or 'year'"),
):
    """Retrieve catalog movies with pagination, searching, and filtering."""
    return movie_service.get_movies(page=page, limit=limit, query=q, genre=genre, sort_by=sort_by)


@router.get("/genres", response_model=List[str])
def list_genres():
    """Retrieve all standard MovieLens genre categories."""
    return movie_service.all_genres


@router.get("/{movie_id}", response_model=MovieDetail)
def get_movie(movie_id: int):
    """Retrieve comprehensive details, rating distribution, and similar movies for a movie."""
    detail = movie_service.get_movie_detail(movie_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Movie with ID {movie_id} not found in catalog.")
    return detail
