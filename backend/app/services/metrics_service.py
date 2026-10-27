"""Offline Benchmark Metrics Service for FastAPI Backend."""

import pandas as pd
from backend.app.core.config import settings
from backend.app.schemas.metrics import MetricsSummaryResponse, ModelMetric
from app.model_loader import MODEL_REGISTRY


class MetricsService:
    """Loads and formats offline evaluation benchmarks from results/phase4c_metrics.csv."""

    def __init__(self) -> None:
        metrics_csv_path = settings.RESULTS_PATH / "phase4c_metrics.csv"
        if metrics_csv_path.exists():
            self.metrics_df = pd.read_csv(metrics_csv_path)
        else:
            self.metrics_df = pd.DataFrame()

        # Map display model names to model_id
        self.name_to_id = {meta["name"]: meta["id"] for meta in MODEL_REGISTRY.values()}
        # Add fallback variants
        self.name_to_id["Popularity"] = "popularity"
        self.name_to_id["Content-Based"] = "content_based"
        self.name_to_id["User-Based CF"] = "user_cf"
        self.name_to_id["Item-Based CF"] = "item_cf"
        self.name_to_id["KNNBaseline"] = "knn_baseline"
        self.name_to_id["SVD"] = "svd"
        self.name_to_id["Funk SVD"] = "svd"
        self.name_to_id["SVD++"] = "svdpp"
        self.name_to_id["NMF"] = "nmf"
        self.name_to_id["Neural CF"] = "neural_cf"
        self.name_to_id["Neural CF (NeuMF)"] = "neural_cf"
        self.name_to_id["Hybrid"] = "hybrid"
        self.name_to_id["Hybrid (CF + CB)"] = "hybrid"

    def get_metrics_summary(self) -> MetricsSummaryResponse:
        if self.metrics_df.empty:
            return MetricsSummaryResponse(
                total_models=0,
                best_hit_rate_model="N/A",
                best_ndcg_model="N/A",
                best_precision_model="N/A",
                metrics=[],
            )

        best_hr_row = self.metrics_df.loc[self.metrics_df["hit_rate_at_10"].idxmax()]
        best_ndcg_row = self.metrics_df.loc[self.metrics_df["ndcg_at_10"].idxmax()]
        best_p_row = self.metrics_df.loc[self.metrics_df["precision_at_10"].idxmax()]

        metrics_list = []
        for _, row in self.metrics_df.iterrows():
            m_name = str(row["model"])
            m_id = self.name_to_id.get(m_name, m_name.lower().replace(" ", "_"))
            metrics_list.append(ModelMetric(
                model=m_name,
                model_id=m_id,
                precision_at_10=round(float(row["precision_at_10"]), 4),
                recall_at_10=round(float(row["recall_at_10"]), 4),
                f1_at_10=round(float(row["f1_at_10"]), 4),
                ndcg_at_10=round(float(row["ndcg_at_10"]), 4),
                hit_rate_at_10=round(float(row["hit_rate_at_10"]), 4),
                evaluation_users=int(row.get("evaluation_users", 935)),
                fit_time_seconds=round(float(row.get("fit_time_seconds", 0.0)), 2),
                evaluate_time_seconds=round(float(row.get("evaluate_time_seconds", 0.0)), 2),
                total_time_seconds=round(float(row.get("total_time_seconds", 0.0)), 2),
            ))

        return MetricsSummaryResponse(
            total_models=len(metrics_list),
            best_hit_rate_model=str(best_hr_row["model"]),
            best_ndcg_model=str(best_ndcg_row["model"]),
            best_precision_model=str(best_p_row["model"]),
            metrics=metrics_list,
        )


metrics_service = MetricsService()
