"""Pydantic Schemas for Offline Evaluation Metrics."""

from typing import List
from pydantic import BaseModel


class ModelMetric(BaseModel):
    model: str
    model_id: str
    precision_at_10: float
    recall_at_10: float
    f1_at_10: float
    ndcg_at_10: float
    hit_rate_at_10: float
    evaluation_users: int
    fit_time_seconds: float
    evaluate_time_seconds: float
    total_time_seconds: float


class MetricsSummaryResponse(BaseModel):
    total_models: int
    best_hit_rate_model: str
    best_ndcg_model: str
    best_precision_model: str
    metrics: List[ModelMetric]
