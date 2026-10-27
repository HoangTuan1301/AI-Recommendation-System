"""Pydantic Schemas for AI Recommendation Models."""

from typing import List
from pydantic import BaseModel


class ModelInfo(BaseModel):
    id: str
    name: str
    family: str
    score_type: str
    score_range: str
    score_format: str
    description: str
    complexity: str


class ModelsListResponse(BaseModel):
    total: int
    models: List[ModelInfo]
