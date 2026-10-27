"""Pydantic Schemas for Movie Entities."""

from typing import List, Optional
from pydantic import BaseModel, Field


class MovieSummary(BaseModel):
    movie_id: int
    title: str
    year: Optional[int] = None
    release_date: Optional[str] = None
    genres: List[str] = Field(default_factory=list)
    genres_str: str = ""
    avg_rating: Optional[float] = None
    rating_count: int = 0
    poster_url: Optional[str] = None


class MovieDetail(MovieSummary):
    imdb_url: Optional[str] = None
    rating_distribution: Optional[dict] = None
    similar_movies: List[MovieSummary] = Field(default_factory=list)


class MoviePagination(BaseModel):
    total: int
    page: int
    limit: int
    items: List[MovieSummary]
