"""Evaluation Metrics Endpoint."""

from fastapi import APIRouter
from backend.app.schemas.metrics import MetricsSummaryResponse
from backend.app.services.metrics_service import metrics_service

router = APIRouter(prefix="/metrics", tags=["Metrics"])


@router.get("", response_model=MetricsSummaryResponse)
def get_metrics():
    """Retrieve official offline benchmark evaluation results for all 10 models."""
    return metrics_service.get_metrics_summary()
