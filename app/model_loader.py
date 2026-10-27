"""Model Registry, Loading, and Checkpoint Manager for Streamlit Demo.

Provides unified factory methods, checkpoint caching, and metadata descriptions
for all 10 recommendation models on MovieLens 100K:
1. Popularity Baseline
2. Content-Based Recommender
3. User-Based Collaborative Filtering
4. Item-Based Collaborative Filtering
5. KNN with Baselines (KNNBaseline)
6. Funk SVD (Matrix Factorization)
7. SVD++ (Factorization with Implicit Feedback)
8. NMF (Non-Negative Matrix Factorization)
9. Neural CF (NeuMF)
10. Weighted Hybrid Recommender
"""

from pathlib import Path
import sys
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.base_recommender import BaseRecommender
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

logger = logging.getLogger(__name__)

CHECKPOINTS_DIR = REPO_ROOT / "models" / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "Popularity": {
        "id": "popularity",
        "name": "Popularity Baseline",
        "family": "Baseline",
        "score_type": "IMDb Weighted Rating",
        "score_range": "1.0 - 5.0 sao",
        "score_format": "{:.2f} ★",
        "description": "Ước lượng Bayes độ hài lòng tổng thể kết hợp giữa điểm trung bình và số lượng đánh giá (với m = 25 lượt). Không cá nhân hóa.",
        "complexity": "O(1) inference",
    },
    "Content-Based": {
        "id": "content_based",
        "name": "Content-Based Recommender",
        "family": "Content-Based",
        "score_type": "TF-IDF Cosine Similarity",
        "score_range": "0.0 - 1.0",
        "score_format": "{:.3f} sim",
        "description": "Biểu diễn thể loại và tiêu đề phim qua TF-IDF (2,350 chiều), xây dựng hồ sơ sở thích người dùng và đo tương đồng Cosine.",
        "complexity": "O(d) inference",
    },
    "User-Based CF": {
        "id": "user_cf",
        "name": "User-Based Collaborative Filtering",
        "family": "Neighborhood CF",
        "score_type": "Predicted Rating",
        "score_range": "1.0 - 5.0 sao",
        "score_format": "{:.2f} ★",
        "description": "Tìm K láng giềng có thị hiếu tương đồng qua tương quan Pearson thu hẹp (shrinkage gamma = 5.0, K = 40) và tổng hợp độ lệch.",
        "complexity": "O(K) inference",
    },
    "Item-Based CF": {
        "id": "item_cf",
        "name": "Item-Based Collaborative Filtering",
        "family": "Neighborhood CF",
        "score_type": "Predicted Rating",
        "score_range": "1.0 - 5.0 sao",
        "score_format": "{:.2f} ★",
        "description": "Tính toán độ tương đồng Adjusted Cosine giữa các cặp phim (K = 30) dựa trên rating đã trừ trung bình người dùng.",
        "complexity": "O(K) inference",
    },
    "KNNBaseline": {
        "id": "knn_baseline",
        "name": "KNN with Baselines",
        "family": "Neighborhood CF",
        "score_type": "Predicted Rating",
        "score_range": "1.0 - 5.0 sao",
        "score_format": "{:.2f} ★",
        "description": "Kết hợp mô hình xu hướng nền (Baseline Predictor: user & item biases) với tương quan Pearson trên phần dư (residuals).",
        "complexity": "O(K) inference",
    },
    "Funk SVD": {
        "id": "svd",
        "name": "Funk SVD (Matrix Factorization)",
        "family": "Matrix Factorization",
        "score_type": "Predicted Rating",
        "score_range": "1.0 - 5.0 sao",
        "score_format": "{:.2f} ★",
        "description": "Phân rã ma trận rating sang không gian ẩn 20 chiều, tối ưu hóa bằng Mini-Batch SGD với L2 Regularization.",
        "complexity": "O(k) dot product",
    },
    "SVD++": {
        "id": "svdpp",
        "name": "SVD++ (Factorization + Implicit)",
        "family": "Matrix Factorization",
        "score_type": "Predicted Rating",
        "score_range": "1.0 - 5.0 sao",
        "score_format": "{:.2f} ★",
        "description": "Korens SVD++ tích hợp phản hồi ngầm định (implicit feedback) từ toàn bộ danh sách phim người dùng đã từng xem.",
        "complexity": "O(k + |I_u|) dot product",
    },
    "NMF": {
        "id": "nmf",
        "name": "Non-Negative Matrix Factorization (NMF)",
        "family": "Matrix Factorization",
        "score_type": "Predicted Rating",
        "score_range": "1.0 - 5.0 sao",
        "score_format": "{:.2f} ★",
        "description": "Phân rã ma trận với ràng buộc không âm (50 nhân tố ẩn), các đặc trưng ẩn mang tính chất thành phần có thể giải thích trực tiếp.",
        "complexity": "O(k) dot product",
    },
    "Neural CF (NeuMF)": {
        "id": "neural_cf",
        "name": "Neural CF (Neural Matrix Factorization)",
        "family": "Deep Learning",
        "score_type": "Interaction Likelihood",
        "score_range": "0.0 - 1.0 (Xác suất)",
        "score_format": "{:.3f} prob",
        "description": "Kiến trúc sâu kết hợp GMF (tương tác tuyến tính) và MLP (tương tác phi tuyến [128, 64, 32]), tối ưu qua BCEWithLogitsLoss.",
        "complexity": "Deep Forward Pass",
    },
    "Hybrid (CF + CB)": {
        "id": "hybrid",
        "name": "Weighted Hybrid Recommender",
        "family": "Hybrid Recommender",
        "score_type": "Score-Normalized Fusion",
        "score_range": "0.0 - 1.0 (Điểm tổng hợp)",
        "score_format": "{:.3f} score",
        "description": "Chuẩn hóa động Min-Max trên tập ứng viên và kết hợp trọng số: 80% User-Based CF + 20% Content-Based.",
        "complexity": "CF + CB Ensemble",
    },
}


def build_raw_model(model_key: str, movies_df: pd.DataFrame) -> BaseRecommender:
    """Instantiate a model with its optimal hyperparameter configuration."""
    if model_key == "Popularity":
        return PopularityRecommender(name="Popularity", method="weighted_rating")

    elif model_key == "Content-Based":
        return ContentBasedRecommender(movies_df=movies_df, name="Content-Based")

    elif model_key == "User-Based CF":
        return UserBasedCF(name="User-Based CF", k_neighbors=40)

    elif model_key == "Item-Based CF":
        return ItemBasedCF(name="Item-Based CF", k_neighbors=30)

    elif model_key == "KNNBaseline":
        return KNNBaseline(
            name="KNNBaseline",
            k_neighbors=30,
            lambda_user=10.0,
            lambda_item=25.0,
            lambda_shrink=100.0,
        )

    elif model_key == "Funk SVD":
        return SVDRecommender(
            n_factors=20,
            learning_rate=0.010,
            regularization=0.05,
            n_epochs=30,
            batch_size=512,
            random_state=42,
            early_stopping=False,
            name="Funk SVD",
        )

    elif model_key == "SVD++":
        return SVDppRecommender(
            n_factors=20,
            learning_rate=0.010,
            regularization=0.05,
            n_epochs=20,
            random_state=42,
            early_stopping=False,
            name="SVD++",
        )

    elif model_key == "NMF":
        return NMFRecommender(
            n_factors=50,
            learning_rate=0.005,
            regularization=0.02,
            n_epochs=30,
            batch_size=512,
            random_state=42,
            early_stopping=False,
            name="NMF",
        )

    elif model_key == "Neural CF (NeuMF)":
        return NeuralCFRecommender(
            n_factors_gmf=32,
            n_factors_mlp=64,
            layers=[128, 64, 32],
            learning_rate=0.001,
            neg_ratio=4,
            n_epochs=15,
            batch_size=512,
            random_state=42,
            name="Neural CF",
        )

    elif model_key == "Hybrid (CF + CB)":
        user_cf = UserBasedCF(name="User-Based CF", k_neighbors=40)
        cb_model = ContentBasedRecommender(movies_df=movies_df, name="Content-Based")
        return HybridRecommender(
            cf_model=user_cf,
            cb_model=cb_model,
            alpha=0.80,
            normalization="min_max",
            name="Hybrid",
        )

    else:
        raise ValueError(f"Unknown model key: '{model_key}'. Available: {list(MODEL_REGISTRY.keys())}")


def get_checkpoint_path(model_key: str) -> Path:
    """Return the filesystem path for a model's serialized checkpoint."""
    slug = MODEL_REGISTRY[model_key]["id"]
    return CHECKPOINTS_DIR / f"{slug}.pkl"


def load_or_train_model(
    model_key: str,
    train_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    force_retrain: bool = False,
) -> BaseRecommender:
    """Load a trained model from checkpoint if available; otherwise train, save, and return.

    Args:
        model_key: Registry key for the model.
        train_df: Clean training DataFrame.
        movies_df: Clean movies DataFrame.
        force_retrain: If True, bypass existing checkpoint and retrain.

    Returns:
        BaseRecommender: Fitted recommender ready for inference.
    """
    ckpt_path = get_checkpoint_path(model_key)

    if ckpt_path.exists() and not force_retrain:
        try:
            logger.info("Loading model '%s' from checkpoint: %s", model_key, ckpt_path)
            model = joblib.load(ckpt_path)
            # Ensure model flag is active
            model.is_fitted = True
            return model
        except Exception as exc:
            logger.warning("Failed to load checkpoint for '%s' (%s). Retraining...", model_key, exc)

    logger.info("Training model '%s' from scratch...", model_key)
    model = build_raw_model(model_key, movies_df=movies_df)
    model.fit(train_df)

    try:
        joblib.dump(model, ckpt_path)
        logger.info("Saved checkpoint for '%s' to %s", model_key, ckpt_path)
    except Exception as exc:
        logger.warning("Could not persist checkpoint for '%s': %s", model_key, exc)

    return model


def get_model_recommendations_with_scores(
    model: BaseRecommender,
    model_key: str,
    user_id: int,
    top_n: int = 10,
    candidate_items: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """Generate recommendations and compute individual prediction scores for display.

    Args:
        model: Fitted recommender instance.
        model_key: Registry key for formatting metadata.
        user_id: Target user ID.
        top_n: Number of recommendations (5, 10, 20).
        candidate_items: Candidates to rank. If None, all unobserved items.

    Returns:
        List[Dict[str, Any]]: List of recommended item dicts with keys:
            ['rank', 'movie_id', 'raw_score', 'formatted_score', 'score_type']
    """
    rec_ids = model.recommend(user_id=user_id, top_n=top_n, candidate_items=candidate_items)
    meta = MODEL_REGISTRY.get(model_key, {})
    fmt = meta.get("score_format", "{:.2f}")
    stype = meta.get("score_type", "Score")

    results = []
    for rank, mid in enumerate(rec_ids, start=1):
        raw_sc = 0.0
        try:
            raw_sc = float(model.predict(user_id, mid))
        except Exception:
            raw_sc = 0.0

        try:
            fmt_sc = fmt.format(raw_sc)
        except Exception:
            fmt_sc = f"{raw_sc:.2f}"

        results.append({
            "rank": rank,
            "movie_id": int(mid),
            "raw_score": raw_sc,
            "formatted_score": fmt_sc,
            "score_type": stype,
        })

    return results
