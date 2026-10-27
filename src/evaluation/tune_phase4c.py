"""Validation Tuning Harness for Phase 4C: Neural CF and Hybrid Models.

Strict Isolation Protocol:
- Uses ONLY train_inner.csv (64,005 rows) and val.csv (15,995 rows).
- test.csv is strictly forbidden and never accessed.
- Evaluates candidate parameter configurations on 927 validation users.
- Outputs complete tuning logs to results/phase4c_tuning_results.csv.
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
from src.models.content_based import ContentBasedRecommender
from src.models.user_cf import UserBasedCF
from src.models.item_cf import ItemBasedCF
from src.models.svdpp import SVDppRecommender
from src.models.neural_cf import NeuralCFRecommender
from src.models.hybrid import HybridRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_tuning_phase4c() -> pd.DataFrame:
    """Run systematic parameter sweep for Neural CF and Hybrid on the Validation set."""
    train_inner_df, val_df, _ = get_or_create_validation_split()
    movies_df = pd.read_csv(REPO_ROOT / "data" / "processed" / "movies_clean.csv")
    all_item_ids = movies_df["movie_id"].unique()

    val_evaluator = TopNEvaluator(
        train_df=train_inner_df,
        test_df=val_df,
        all_item_ids=all_item_ids,
        relevance_threshold=4.0,
        k=10,
    )

    tuning_records: List[Dict[str, Any]] = []

    # =========================================================================
    # PART 1: NEURAL COLLABORATIVE FILTERING (NeuMF) TUNING
    # =========================================================================
    ncf_configs = [
        {"desc": "NCF_small", "gmf": 8, "mlp": 16, "layers": (32, 16), "lr": 0.001, "neg": 2, "epochs": 15},
        {"desc": "NCF_standard", "gmf": 16, "mlp": 32, "layers": (64, 32, 16), "lr": 0.001, "neg": 4, "epochs": 15},
        {"desc": "NCF_high_lr", "gmf": 16, "mlp": 32, "layers": (64, 32, 16), "lr": 0.002, "neg": 4, "epochs": 15},
        {"desc": "NCF_large", "gmf": 32, "mlp": 64, "layers": (128, 64, 32), "lr": 0.001, "neg": 4, "epochs": 15},
    ]

    for cfg in ncf_configs:
        logger.info("Evaluating NCF config: %s (gmf=%d, mlp=%d, lr=%.4f, neg=%d)...", cfg["desc"], cfg["gmf"], cfg["mlp"], cfg["lr"], cfg["neg"])
        model = NeuralCFRecommender(
            n_factors_gmf=cfg["gmf"],
            n_factors_mlp=cfg["mlp"],
            layers=cfg["layers"],
            learning_rate=cfg["lr"],
            neg_ratio=cfg["neg"],
            n_epochs=cfg["epochs"],
            batch_size=512,
            random_state=42,
            name=f"NeuralCF_{cfg['desc']}",
        )
        t_fit = time.time()
        model.fit(train_inner_df)
        fit_time = time.time() - t_fit

        t_eval = time.time()
        metrics = val_evaluator.evaluate_model(model)
        eval_time = time.time() - t_eval

        rec = {
            "model_family": "Neural CF",
            "config_name": cfg["desc"],
            "parameters": f"gmf={cfg['gmf']}_mlp={cfg['mlp']}_layers={list(cfg['layers'])}_lr={cfg['lr']}_neg={cfg['neg']}_ep={cfg['epochs']}",
            "val_precision_at_10": metrics["precision_at_10"],
            "val_recall_at_10": metrics["recall_at_10"],
            "val_f1_at_10": metrics["f1_at_10"],
            "val_ndcg_at_10": metrics["ndcg_at_10"],
            "val_hit_rate_at_10": metrics["hit_rate_at_10"],
            "fit_time_seconds": round(fit_time, 4),
            "eval_time_seconds": round(eval_time, 4),
        }
        tuning_records.append(rec)
        logger.info("Result %s -> Val HR@10=%.4f, Val NDCG@10=%.4f", cfg["desc"], rec["val_hit_rate_at_10"], rec["val_ndcg_at_10"])

    # =========================================================================
    # PART 2: HYBRID RECOMMENDER TUNING
    # =========================================================================
    # Pre-fit shared sub-models on train_inner to ensure zero overhead
    logger.info("Pre-fitting sub-models for Hybrid tuning on train_inner...")
    user_cf = UserBasedCF(name="User-Based CF", k_neighbors=40).fit(train_inner_df)
    item_cf = ItemBasedCF(name="Item-Based CF", k_neighbors=30).fit(train_inner_df)
    svd_pp = SVDppRecommender(name="SVD++", n_factors=20, learning_rate=0.01, regularization=0.05, n_epochs=20, random_state=42).fit(train_inner_df)
    cb_model = ContentBasedRecommender(movies_df=movies_df, name="Content-Based").fit(train_inner_df)

    hybrid_configs = [
        {"cf_model": user_cf, "cf_name": "UserCF", "alpha": 0.5, "norm": "min_max"},
        {"cf_model": user_cf, "cf_name": "UserCF", "alpha": 0.7, "norm": "min_max"},
        {"cf_model": user_cf, "cf_name": "UserCF", "alpha": 0.8, "norm": "min_max"},
        {"cf_model": user_cf, "cf_name": "UserCF", "alpha": 0.85, "norm": "min_max"},
        {"cf_model": user_cf, "cf_name": "UserCF", "alpha": 0.9, "norm": "min_max"},
        {"cf_model": user_cf, "cf_name": "UserCF", "alpha": 0.8, "norm": "z_score"},
        {"cf_model": item_cf, "cf_name": "ItemCF", "alpha": 0.8, "norm": "min_max"},
        {"cf_model": svd_pp, "cf_name": "SVDpp", "alpha": 0.8, "norm": "min_max"},
    ]

    for cfg in hybrid_configs:
        c_name = f"Hybrid_{cfg['cf_name']}_a{int(cfg['alpha']*100)}_{cfg['norm']}"
        logger.info("Evaluating Hybrid config: %s...", c_name)

        model = HybridRecommender(
            cf_model=cfg["cf_model"],
            cb_model=cb_model,
            alpha=cfg["alpha"],
            normalization=cfg["norm"],
            name=c_name,
        )
        # Components are pre-fitted
        model._store_train_metadata(train_inner_df)
        model.is_fitted = True

        t_eval = time.time()
        metrics = val_evaluator.evaluate_model(model)
        eval_time = time.time() - t_eval

        rec = {
            "model_family": "Hybrid",
            "config_name": c_name,
            "parameters": f"cf={cfg['cf_name']}_alpha={cfg['alpha']}_norm={cfg['norm']}",
            "val_precision_at_10": metrics["precision_at_10"],
            "val_recall_at_10": metrics["recall_at_10"],
            "val_f1_at_10": metrics["f1_at_10"],
            "val_ndcg_at_10": metrics["ndcg_at_10"],
            "val_hit_rate_at_10": metrics["hit_rate_at_10"],
            "fit_time_seconds": 0.0,
            "eval_time_seconds": round(eval_time, 4),
        }
        tuning_records.append(rec)
        logger.info("Result %s -> Val HR@10=%.4f, Val NDCG@10=%.4f", c_name, rec["val_hit_rate_at_10"], rec["val_ndcg_at_10"])

    out_df = pd.DataFrame(tuning_records)
    out_path = RESULTS_DIR / "phase4c_tuning_results.csv"
    out_df.to_csv(out_path, index=False)
    logger.info("Phase 4C tuning results successfully saved to %s", out_path)
    return out_df


if __name__ == "__main__":
    run_tuning_phase4c()
