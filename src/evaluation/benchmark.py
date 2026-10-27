"""Benchmark script for Phase 3 baseline models (Popularity & Content-Based).

Executes a unified evaluation run using TopNEvaluator on the held-out Test set.
Saves official results to results/phase3_baseline_metrics.csv.
"""

from pathlib import Path
import sys

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import logging
import pandas as pd

from src.evaluation.evaluator import TopNEvaluator
from src.models.popularity import PopularityRecommender
from src.models.content_based import ContentBasedRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_phase3_benchmark() -> pd.DataFrame:
    """Run benchmark experiments on baseline models and export metrics CSV."""
    logger.info("Loading processed datasets from %s...", PROCESSED_DIR)
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    movies_df = pd.read_csv(PROCESSED_DIR / "movies_clean.csv")

    # 1. Initialize Evaluator
    evaluator = TopNEvaluator(
        train_df=train_df,
        test_df=test_df,
        all_item_ids=movies_df["movie_id"].unique(),
        relevance_threshold=4.0,
        k=10,
    )

    results = []

    # 2. Benchmark Model 1: Popularity Baseline
    logger.info("--- [1/2] Benchmarking PopularityRecommender ---")
    pop_model = PopularityRecommender(name="Popularity_Baseline", method="weighted_rating")
    pop_model.fit(train_df)
    pop_metrics = evaluator.evaluate_model(pop_model)
    results.append(pop_metrics)

    # 3. Benchmark Model 2: Content-Based Baseline
    logger.info("--- [2/2] Benchmarking ContentBasedRecommender ---")
    cb_model = ContentBasedRecommender(movies_df=movies_df, name="ContentBased_Baseline")
    cb_model.fit(train_df)
    cb_metrics = evaluator.evaluate_model(cb_model)
    results.append(cb_metrics)

    # 4. Compile and validate DataFrame
    results_df = pd.DataFrame(results)

    # Validation: all ranking metrics must be strictly in [0.0, 1.0]
    metric_cols = ["precision_at_10", "recall_at_10", "f1_at_10", "ndcg_at_10", "hit_rate_at_10"]
    for col in metric_cols:
        assert (results_df[col] >= 0.0).all() and (results_df[col] <= 1.0).all(), f"Metric {col} out of range [0, 1]!"

    # 5. Save results to CSV
    output_csv = RESULTS_DIR / "phase3_baseline_metrics.csv"
    results_df.to_csv(output_csv, index=False, encoding="utf-8")
    logger.info("Benchmark metrics successfully written to: %s", output_csv.resolve())

    return results_df


if __name__ == "__main__":
    df = run_phase3_benchmark()
    print("\n=================== PHASE 3 BENCHMARK RESULTS ===================")
    print(df.to_string(index=False))
    print("=================================================================\n")
