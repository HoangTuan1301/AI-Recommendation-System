"""Benchmark execution for Phase 4C: 10 Recommendation Models.

Evaluates all 10 models on MovieLens 100K under strictly identical evaluation conditions:
1. Popularity Baseline (IMDb Bayesian Average)
2. Content-Based Baseline (TF-IDF & Cosine Similarity)
3. User-Based Collaborative Filtering (User-User KNN)
4. Item-Based Collaborative Filtering (Item-Item KNN)
5. KNN with Baselines (KNNBaseline)
6. Funk SVD (Matrix Factorization)
7. SVD++ (Factorization with Implicit Feedback)
8. NMF (Non-Negative Matrix Factorization)
9. Neural CF (Neural Matrix Factorization - NeuMF)
10. Hybrid Recommender (Score-Normalized Weighted Fusion)

Protocol:
- Held-out Test set (935 evaluation users with rating >= 4.0 in Test)
- Cutoff K = 10
- Relevant threshold rating >= 4.0
- Candidate set: all catalog items minus user's training history
- Macro-averaging over evaluated users
- Detailed recording of fit_time_seconds, evaluate_time_seconds, total_time_seconds
- Output saved to results/phase4c_metrics.csv
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Any, Dict, List, Optional, Sequence, Tuple
import logging
import time
import pandas as pd

from src.evaluation.evaluator import TopNEvaluator
from src.models.popularity import PopularityRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.user_cf import UserBasedCF
from src.models.item_cf import ItemBasedCF
from src.models.knn_baseline import KNNBaseline
from src.models.svd import SVDRecommender
from src.models.svdpp import SVDppRecommender
from src.models.nmf import NMFRecommender
from src.models.neural_cf import NeuralCFRecommender
from src.models.hybrid import HybridRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_models(
    movies_df: pd.DataFrame,
    ncf_params: Optional[Dict[str, Any]] = None,
    hybrid_alpha: float = 0.80,
    hybrid_norm: str = "min_max",
) -> List[Any]:
    """Instantiate all 10 benchmark recommendation models.

    Args:
        movies_df: Clean movies DataFrame metadata.
        ncf_params: Selected hyperparameters for Neural CF from Validation.
        hybrid_alpha: Selected fusion weight alpha from Validation.
        hybrid_norm: Score normalization method ('min_max' or 'z_score').

    Returns:
        List[BaseRecommender]: The 10 models ready for fitting and evaluation.
    """
    if ncf_params is None:
        ncf_params = {
            "n_factors_gmf": 32,
            "n_factors_mlp": 64,
            "layers": [128, 64, 32],
            "learning_rate": 0.001,
            "neg_ratio": 4,
            "n_epochs": 15,
        }

    cb_base = ContentBasedRecommender(movies_df=movies_df, name="Content-Based")
    user_cf_base = UserBasedCF(name="User-Based CF", k_neighbors=40)

    # Hybrid component reuses UserBasedCF and ContentBasedRecommender
    hybrid_model = HybridRecommender(
        cf_model=user_cf_base,
        cb_model=cb_base,
        alpha=hybrid_alpha,
        normalization=hybrid_norm,
        name="Hybrid",
    )

    models = [
        # Baselines (Phase 3)
        PopularityRecommender(name="Popularity", method="weighted_rating"),
        cb_base,
        # Neighborhood-based Collaborative Filtering (Phase 4A)
        user_cf_base,
        ItemBasedCF(name="Item-Based CF", k_neighbors=30),
        KNNBaseline(
            name="KNNBaseline",
            k_neighbors=30,
            lambda_user=10.0,
            lambda_item=25.0,
            lambda_shrink=100.0,
        ),
        # Matrix Factorization Recommenders (Phase 4B)
        SVDRecommender(
            n_factors=20,
            learning_rate=0.010,
            regularization=0.05,
            n_epochs=30,
            batch_size=512,
            random_state=42,
            early_stopping=False,
            name="SVD",
        ),
        SVDppRecommender(
            n_factors=20,
            learning_rate=0.010,
            regularization=0.05,
            n_epochs=20,
            random_state=42,
            early_stopping=False,
            name="SVD++",
        ),
        NMFRecommender(
            n_factors=50,
            learning_rate=0.005,
            regularization=0.02,
            n_epochs=30,
            batch_size=512,
            random_state=42,
            early_stopping=False,
            name="NMF",
        ),
        # Deep Learning & Neural Collaborative Filtering (Phase 4C)
        NeuralCFRecommender(
            n_factors_gmf=ncf_params["n_factors_gmf"],
            n_factors_mlp=ncf_params["n_factors_mlp"],
            layers=ncf_params["layers"],
            learning_rate=ncf_params["learning_rate"],
            neg_ratio=ncf_params["neg_ratio"],
            n_epochs=ncf_params["n_epochs"],
            batch_size=512,
            random_state=42,
            name="Neural CF",
        ),
        # Weighted Hybrid Recommender (Phase 4C)
        hybrid_model,
    ]
    return models


def run_phase4c_benchmark(
    ncf_params: Optional[Dict[str, Any]] = None,
    hybrid_alpha: float = 0.80,
    hybrid_norm: str = "min_max",
) -> pd.DataFrame:
    """Execute benchmark experiments across all 10 models and export results."""
    logger.info("Loading processed datasets from %s...", DATA_PROCESSED_DIR)
    train_df = pd.read_csv(DATA_PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(DATA_PROCESSED_DIR / "test.csv")
    movies_df = pd.read_csv(DATA_PROCESSED_DIR / "movies_clean.csv")

    evaluator = TopNEvaluator(
        train_df=train_df,
        test_df=test_df,
        all_item_ids=movies_df["movie_id"].unique(),
        relevance_threshold=4.0,
        k=10,
    )

    models = build_models(
        movies_df=movies_df,
        ncf_params=ncf_params,
        hybrid_alpha=hybrid_alpha,
        hybrid_norm=hybrid_norm,
    )

    results_records: List[Dict[str, Any]] = []
    logger.info("Starting Phase 4C Benchmark Evaluation across %d models...", len(models))

    for model in models:
        mname = model.name
        logger.info("==================================================")
        logger.info("Evaluating model: %s", mname)

        # 1. Fit timing
        t_fit_start = time.time()
        # For Hybrid, if sub-models are already fitted, fit() reuses them seamlessly
        model.fit(train_df)
        fit_time = time.time() - t_fit_start
        logger.info("Model %s fitted in %.4f seconds.", mname, fit_time)

        # 2. Evaluation timing
        t_eval_start = time.time()
        metrics = evaluator.evaluate_model(model, model_name=mname)
        eval_time = time.time() - t_eval_start
        logger.info("Model %s evaluated in %.4f seconds.", mname, eval_time)

        row = {
            "model": mname,
            "precision_at_10": metrics["precision_at_10"],
            "recall_at_10": metrics["recall_at_10"],
            "f1_at_10": metrics["f1_at_10"],
            "ndcg_at_10": metrics["ndcg_at_10"],
            "hit_rate_at_10": metrics["hit_rate_at_10"],
            "evaluation_users": metrics["evaluation_users"],
            "fit_time_seconds": round(fit_time, 4),
            "evaluate_time_seconds": round(eval_time, 4),
            "total_time_seconds": round(fit_time + eval_time, 4),
        }
        results_records.append(row)

    benchmark_df = pd.DataFrame(results_records)
    output_path = RESULTS_DIR / "phase4c_metrics.csv"
    benchmark_df.to_csv(output_path, index=False)
    logger.info("Phase 4C Benchmark successfully exported to %s", output_path)

    print("\n================== PHASE 4C BENCHMARK SUMMARY (10 MODELS) ==================")
    print(
        benchmark_df[
            [
                "model",
                "precision_at_10",
                "recall_at_10",
                "f1_at_10",
                "ndcg_at_10",
                "hit_rate_at_10",
                "total_time_seconds",
            ]
        ].to_string(index=False)
    )
    print("============================================================================\n")
    return benchmark_df


if __name__ == "__main__":
    run_phase4c_benchmark()
