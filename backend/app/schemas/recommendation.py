"""Pydantic Schemas for Recommendations."""

from typing import List, Optional
from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    user_id: int = Field(..., ge=1, le=943, description="Target User ID in MovieLens 100K")
    model_id: str = Field(default="neural_cf", description="Model identifier slug")
    top_n: int = Field(default=10, ge=1, le=50, description="Number of recommendations")


class RecommendedMovie(BaseModel):
    movie_id: int
    title: str
    year: Optional[int] = None
    genres: List[str] = Field(default_factory=list)
    genres_str: str = ""
    rank: int
    raw_score: Optional[float] = None
    formatted_score: str
    score_type: str
    poster_url: Optional[str] = None


class RecommendationResponse(BaseModel):
    user_id: int
    model_id: str
    model_name: str
    top_n: int
    count: int
    inference_time_ms: float
    recommendations: List[RecommendedMovie]
