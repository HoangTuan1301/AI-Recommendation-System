"""Pydantic Schemas for Users and Profiles."""

from typing import Dict, List, Optional
from pydantic import BaseModel
from backend.app.schemas.movie import MovieSummary


class UserSummary(BaseModel):
    user_id: int
    ratings_count: int
    avg_rating: float
    high_rated_count: int


class GenreAffinity(BaseModel):
    genre: str
    count: int
    percentage: float


class UserProfileResponse(BaseModel):
    user_id: int
    ratings_count: int
    avg_rating: float
    high_rated_count: int
    rating_distribution: Dict[int, int]
    favorite_genres: List[GenreAffinity]
    top_rated_movies: List[MovieSummary]


class UsersListResponse(BaseModel):
    total: int
    users: List[UserSummary]
