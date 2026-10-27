"""Pydantic Schemas for 1v1 Model Battle Comparison."""

from typing import List
from pydantic import BaseModel, Field
from backend.app.schemas.recommendation import RecommendedMovie


class CompareRequest(BaseModel):
    user_id: int = Field(..., ge=1, le=943)
    model_a_id: str = Field(default="neural_cf")
    model_b_id: str = Field(default="user_cf")
    top_n: int = Field(default=10, ge=1, le=50)


class CompareResponse(BaseModel):
    user_id: int
    model_a_id: str
    model_a_name: str
    model_b_id: str
    model_b_name: str
    top_n: int
    jaccard_similarity: float
    overlap_count: int
    recommendations_a: List[RecommendedMovie]
    recommendations_b: List[RecommendedMovie]
    shared_movie_ids: List[int]
