"""Benchmark execution for Phase 4A: 5 Recommendation Models.

Evaluates:
1. Popularity Baseline (IMDb Bayesian Average)
2. Content-Based Baseline (TF-IDF & Cosine Similarity)
3. User-Based Collaborative Filtering (User-User KNN)
4. Item-Based Collaborative Filtering (Item-Item KNN)
5. KNN with Baselines (KNNBaseline)

Enforces exact identical evaluation protocol:
- Held-out Test set (935 evaluation users)
- Cutoff K = 10
- Relevant threshold rating >= 4.0
- Candidate set: all catalog items minus user's training history
- Macro-averaging over evaluated users
- Detailed recording of fit_time_seconds and evaluate_time_seconds
- Output saved to results/phase4a_metrics.csv
"""

from pathlib import Path
import sys

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import logging
import time
import pandas as pd

from src.evaluation.evaluator import TopNEvaluator
from src.models.popularity import PopularityRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.user_cf import UserBasedCF
from src.models.item_cf import ItemBasedCF
from src.models.knn_baseline import KNNBaseline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_phase4a_benchmark() -> pd.DataFrame:
    """Run benchmark experiments on all 5 models and export results/phase4a_metrics.csv."""
    logger.info("Loading processed datasets from %s...", PROCESSED_DIR)
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    movies_df = pd.read_csv(PROCESSED_DIR / "movies_clean.csv")

    evaluator = TopNEvaluator(
        train_df=train_df,
        test_df=test_df,
        all_item_ids=movies_df["movie_id"].unique(),
        relevance_threshold=4.0,
        k=10,
    )

    models_to_evaluate = [
        ("Popularity", PopularityRecommender(name="Popularity", method="weighted_rating")),
        ("Content-Based", ContentBasedRecommender(movies_df=movies_df, name="Content-Based")),
        ("User-Based CF", UserBasedCF(name="User-Based CF", k_neighbors=40)),
        ("Item-Based CF", ItemBasedCF(name="Item-Based CF", k_neighbors=30)),
        ("KNNBaseline", KNNBaseline(name="KNNBaseline", k_neighbors=30, lambda_user=10.0, lambda_item=25.0, lambda_shrink=100.0)),
    ]

    records = []

    for model_name, model in models_to_evaluate:
        logger.info("==================================================")
        logger.info("Evaluating [%s]...", model_name)

        # 1. Measure fit time strictly on Train
        t_fit_start = time.time()
        model.fit(train_df)
        fit_time = time.time() - t_fit_start

        # 2. Measure evaluation time on Test
        t_eval_start = time.time()
        eval_metrics = evaluator.evaluate_model(model, model_name=model_name)
        eval_time = time.time() - t_eval_start

        record = {
            "model": model_name,
            "precision_at_10": eval_metrics["precision_at_10"],
            "recall_at_10": eval_metrics["recall_at_10"],
            "f1_at_10": eval_metrics["f1_at_10"],
            "ndcg_at_10": eval_metrics["ndcg_at_10"],
            "hit_rate_at_10": eval_metrics["hit_rate_at_10"],
            "evaluation_users": eval_metrics["evaluation_users"],
            "fit_time_seconds": round(fit_time, 4),
            "evaluate_time_seconds": round(eval_time, 4),
            "total_time_seconds": round(fit_time + eval_time, 4),
        }
        records.append(record)

    results_df = pd.DataFrame(records)

    # Sanity validations
    metric_cols = ["precision_at_10", "recall_at_10", "f1_at_10", "ndcg_at_10", "hit_rate_at_10"]
    for col in metric_cols:
        assert (results_df[col] >= 0.0).all() and (results_df[col] <= 1.0).all(), f"Metric {col} out of range [0, 1]!"
    assert (results_df["evaluation_users"] == 935).all(), "All models must be evaluated on exactly 935 users!"

    output_csv = RESULTS_DIR / "phase4a_metrics.csv"
    results_df.to_csv(output_csv, index=False, encoding="utf-8")
    logger.info("Phase 4A benchmark successfully saved to: %s", output_csv.resolve())

    return results_df


if __name__ == "__main__":
    df = run_phase4a_benchmark()
    print("\n========================= PHASE 4A BENCHMARK RESULTS (5 MODELS) =========================")
    print(df.to_string(index=False))
    print("=========================================================================================\n")
