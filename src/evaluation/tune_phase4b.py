"""Hyperparameter Tuning on Train-Inner / Validation Split for Phase 4B.

Strict Protocol:
- Uses ONLY train_inner.csv and val.csv.
- test.csv is strictly forbidden and never accessed.
- Evaluates candidate parameter sets for SVD, SVD++, and NMF.
- Evaluates both Validation RMSE and Validation NDCG@10 / Hit Rate@10.
- Outputs tuning report to results/phase4b_tuning_results.csv.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Any, Dict, List
import logging
import time
import pandas as pd

from src.evaluation.evaluator import TopNEvaluator
from src.evaluation.validation import get_or_create_validation_split
from src.models.svd import SVDRecommender
from src.models.svdpp import SVDppRecommender
from src.models.nmf import NMFRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_tuning() -> pd.DataFrame:
    """Execute bounded grid search on Validation set and select best configs."""
    train_inner_df, val_df, meta = get_or_create_validation_split()

    movies_clean = pd.read_csv(REPO_ROOT / "data" / "processed" / "movies_clean.csv")
    all_item_ids = movies_clean["movie_id"].unique()

    # Evaluator on validation set
    val_evaluator = TopNEvaluator(
        train_df=train_inner_df,
        test_df=val_df,
        all_item_ids=all_item_ids,
        relevance_threshold=4.0,
        k=10,
    )

    configs = [
        # SVD candidate configs
        {"model_type": "SVD", "n_factors": 20, "lr": 0.005, "reg": 0.02, "epochs": 30},
        {"model_type": "SVD", "n_factors": 50, "lr": 0.005, "reg": 0.02, "epochs": 30},
        {"model_type": "SVD", "n_factors": 20, "lr": 0.010, "reg": 0.05, "epochs": 30},
        {"model_type": "SVD", "n_factors": 50, "lr": 0.010, "reg": 0.05, "epochs": 30},
        # SVD++ candidate configs
        {"model_type": "SVDpp", "n_factors": 20, "lr": 0.005, "reg": 0.02, "epochs": 20},
        {"model_type": "SVDpp", "n_factors": 50, "lr": 0.005, "reg": 0.02, "epochs": 20},
        {"model_type": "SVDpp", "n_factors": 20, "lr": 0.010, "reg": 0.05, "epochs": 20},
        {"model_type": "SVDpp", "n_factors": 50, "lr": 0.010, "reg": 0.05, "epochs": 20},
        # NMF candidate configs
        {"model_type": "NMF", "n_factors": 20, "lr": 0.005, "reg": 0.02, "epochs": 30},
        {"model_type": "NMF", "n_factors": 50, "lr": 0.005, "reg": 0.02, "epochs": 30},
        {"model_type": "NMF", "n_factors": 20, "lr": 0.010, "reg": 0.05, "epochs": 30},
        {"model_type": "NMF", "n_factors": 50, "lr": 0.010, "reg": 0.05, "epochs": 30},
    ]

    records: List[Dict[str, Any]] = []

    for cfg in configs:
        mtype = cfg["model_type"]
        d = cfg["n_factors"]
        lr = cfg["lr"]
        reg = cfg["reg"]
        epochs = cfg["epochs"]

        logger.info("Tuning %s with n_factors=%d, lr=%.3f, reg=%.3f, epochs=%d...", mtype, d, lr, reg, epochs)

        if mtype == "SVD":
            model = SVDRecommender(
                n_factors=d,
                learning_rate=lr,
                regularization=reg,
                n_epochs=epochs,
                random_state=42,
            )
        elif mtype == "SVDpp":
            model = SVDppRecommender(
                n_factors=d,
                learning_rate=lr,
                regularization=reg,
                n_epochs=epochs,
                random_state=42,
            )
        elif mtype == "NMF":
            model = NMFRecommender(
                n_factors=d,
                learning_rate=lr,
                regularization=reg,
                n_epochs=epochs,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown model_type {mtype}")

        t0 = time.time()
        model.fit(train_inner_df, val_df)
        fit_time = time.time() - t0

        t1 = time.time()
        eval_metrics = val_evaluator.evaluate_model(model)
        eval_time = time.time() - t1

        rec = {
            "model_type": mtype,
            "n_factors": d,
            "lr": lr,
            "reg": reg,
            "epochs": epochs,
            "final_train_rmse": model.train_rmse_history[-1] if hasattr(model, "train_rmse_history") and model.train_rmse_history else None,
            "final_val_rmse": model.val_rmse_history[-1] if hasattr(model, "val_rmse_history") and model.val_rmse_history else None,
            "val_precision_at_10": eval_metrics["precision_at_10"],
            "val_recall_at_10": eval_metrics["recall_at_10"],
            "val_f1_at_10": eval_metrics["f1_at_10"],
            "val_ndcg_at_10": eval_metrics["ndcg_at_10"],
            "val_hit_rate_at_10": eval_metrics["hit_rate_at_10"],
            "fit_time_seconds": round(fit_time, 4),
            "eval_time_seconds": round(eval_time, 4),
        }
        records.append(rec)
        logger.info(
            "Result: %s (d=%d, lr=%.3f, reg=%.3f) -> Val HR@10=%.4f, Val NDCG@10=%.4f, Val RMSE=%.4f",
            mtype,
            d,
            lr,
            reg,
            rec["val_hit_rate_at_10"],
            rec["val_ndcg_at_10"],
            rec["final_val_rmse"] or 0.0,
        )

    tuning_df = pd.DataFrame(records)
    out_path = RESULTS_DIR / "phase4b_tuning_results.csv"
    tuning_df.to_csv(out_path, index=False)
    logger.info("Saved tuning results to %s", out_path)
    return tuning_df


if __name__ == "__main__":
    run_tuning()
