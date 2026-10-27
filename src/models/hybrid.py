"""Weighted Hybrid Recommender System.

This module implements a weighted hybrid recommender combining Collaborative Filtering (CF)
and Content-Based Filtering (CB).

Formulation:
    s_{hybrid}(u, i) = \\alpha \\cdot \\tilde{s}_{CF}(u, i) + (1 - \\alpha) \\cdot \\tilde{s}_{CB}(u, i)

Score Normalization:
    Because CF scores (predicted ratings or neighborhood affinities, e.g. [1.0, 5.0])
    and Content-Based scores (TF-IDF cosine similarities, e.g. [0.0, 1.0]) reside on
    incompatible scales and distributions, candidate scores are dynamically normalized
    across the candidate set before fusion:
    - Min-Max Normalization: \\tilde{s} = (s - s_{min}) / (s_{max} - s_{min} + \\epsilon) \\in [0, 1]
    - Z-Score Normalization:  \\tilde{s} = (s - \\mu) / (\\sigma + \\epsilon)

Protocol:
    - alpha in [0.0, 1.0]: relative weight assigned to the CF component (selected on Validation).
    - Guarantees strict exclusion of seen training items.
    - Preserves deterministic ranking with candidate tie-breaking.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Any, Dict, List, Optional, Sequence, Union
import logging
import numpy as np
import pandas as pd

from src.models.base_recommender import BaseRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.user_cf import UserBasedCF

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = REPO_ROOT / "data" / "processed"


def normalize_vector(
    scores: np.ndarray,
    method: str = "min_max",
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Normalize a 1D array of candidate scores.

    Args:
        scores: 1D array of raw scores.
        method: 'min_max' or 'z_score'.
        epsilon: Small constant to avoid zero-division.

    Returns:
        np.ndarray: Normalized 1D array.
    """
    if len(scores) == 0:
        return scores

    if method == "min_max":
        min_v = float(np.min(scores))
        max_v = float(np.max(scores))
        diff = max_v - min_v
        if diff < epsilon:
            return np.full_like(scores, 0.5, dtype=np.float32)
        return ((scores - min_v) / (diff + epsilon)).astype(np.float32)

    elif method == "z_score":
        mean_v = float(np.mean(scores))
        std_v = float(np.std(scores))
        if std_v < epsilon:
            return np.zeros_like(scores, dtype=np.float32)
        return ((scores - mean_v) / (std_v + epsilon)).astype(np.float32)

    else:
        raise ValueError(f"Unsupported normalization method: '{method}'")


class HybridRecommender(BaseRecommender):
    """Score-normalized Weighted Hybrid Recommender."""

    def __init__(
        self,
        cf_model: Optional[BaseRecommender] = None,
        cb_model: Optional[ContentBasedRecommender] = None,
        alpha: float = 0.8,
        normalization: str = "min_max",
        name: str = "Hybrid",
    ) -> None:
        """Initialize the Hybrid Recommender.

        Args:
            cf_model: Fitted or unfitted Collaborative Filtering recommender.
                      Defaults to UserBasedCF(k_neighbors=40).
            cb_model: Fitted or unfitted ContentBasedRecommender.
                      Defaults to ContentBasedRecommender loaded from movies_clean.csv.
            alpha: Weight assigned to CF model in [0.0, 1.0] (default 0.8).
                   Content-Based receives weight (1 - alpha).
            normalization: Method to normalize component scores ('min_max' or 'z_score').
            name: Model identifier name.
        """
        super().__init__(name=name)
        self.alpha = float(alpha)
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"Alpha must be in range [0.0, 1.0], got {self.alpha}")

        self.normalization = normalization.lower()
        if self.normalization not in ("min_max", "z_score"):
            raise ValueError(f"Invalid normalization '{self.normalization}'. Must be 'min_max' or 'z_score'.")

        self.cf_model: BaseRecommender = cf_model if cf_model is not None else UserBasedCF(name="User-Based CF")

        if cb_model is not None:
            self.cb_model = cb_model
        else:
            movies_clean_path = PROCESSED_DATA_DIR / "movies_clean.csv"
            movies_df = pd.read_csv(movies_clean_path) if movies_clean_path.exists() else pd.DataFrame()
            self.cb_model = ContentBasedRecommender(movies_df=movies_df, name="Content-Based")

    def fit(self, train_df: pd.DataFrame) -> "HybridRecommender":
        """Fit both CF and Content-Based sub-models on the training dataset.

        Args:
            train_df: DataFrame containing ['user_id', 'movie_id', 'rating'].

        Returns:
            self: Fitted Hybrid instance.
        """
        self._store_train_metadata(train_df)

        logger.info(
            "Fitting %s (alpha=%.2f, normalization=%s, CF=%s, CB=%s) on %d rows...",
            self.name,
            self.alpha,
            self.normalization,
            self.cf_model.name,
            self.cb_model.name,
            len(train_df),
        )

        # Fit CF model if not already fitted
        if not self.cf_model.is_fitted:
            logger.info("Fitting sub-model CF: %s...", self.cf_model.name)
            self.cf_model.fit(train_df)

        # Fit CB model if not already fitted
        if not self.cb_model.is_fitted:
            logger.info("Fitting sub-model CB: %s...", self.cb_model.name)
            self.cb_model.fit(train_df)

        self.is_fitted = True
        logger.info("Fitted %s successfully.", self.name)
        return self

    def _get_cf_candidate_scores(self, user_id: int, candidate_list: List[int]) -> np.ndarray:
        """Efficiently extract unnormalized candidate scores from the CF model."""
        if hasattr(self.cf_model, "_score_all_items_for_user"):
            all_scores = self.cf_model._score_all_items_for_user(user_id)
            max_idx = len(all_scores) - 1
            return np.array(
                [all_scores[cid] if 0 <= cid <= max_idx else 0.0 for cid in candidate_list],
                dtype=np.float32,
            )
        elif hasattr(self.cf_model, "user_to_idx") and hasattr(self.cf_model, "P"):
            u_idx = self.cf_model.user_to_idx.get(user_id)
            if u_idx is not None and hasattr(self.cf_model, "bu"):
                user_bias = float(self.cf_model.bu[u_idx])
                user_factor = self.cf_model.P[u_idx]
                scores = []
                for cid in candidate_list:
                    idx = self.cf_model.item_to_idx.get(cid)
                    if idx is not None:
                        sc = self.cf_model.mu + user_bias + self.cf_model.bi[idx] + float(np.dot(self.cf_model.Q[idx], user_factor))
                        scores.append(sc)
                    else:
                        scores.append(self.cf_model.mu + user_bias)
                return np.array(scores, dtype=np.float32)
        # Point-wise fallback
        scores = []
        for cid in candidate_list:
            try:
                scores.append(float(self.cf_model.predict(user_id, cid)))
            except Exception:
                scores.append(0.0)
        return np.array(scores, dtype=np.float32)

    def _get_cb_candidate_scores(self, user_id: int, candidate_list: List[int]) -> np.ndarray:
        """Efficiently extract unnormalized candidate scores from the Content-Based model."""
        if hasattr(self.cb_model, "user_profiles") and self.cb_model.user_profiles is not None:
            if 0 <= user_id <= self.cb_model.max_user_id and user_id in self.cb_model.all_train_users:
                profile = self.cb_model.user_profiles[user_id]
            else:
                profile = self.cb_model.global_fallback_profile
            valid_indices = np.array(
                [cid if 0 <= cid <= self.cb_model.max_movie_id else 0 for cid in candidate_list],
                dtype=np.int64,
            )
            cand_vectors = self.cb_model.item_features[valid_indices]
            return np.dot(cand_vectors, profile).astype(np.float32)

        # Point-wise fallback
        scores = []
        for cid in candidate_list:
            try:
                scores.append(float(self.cb_model.predict(user_id, cid)))
            except Exception:
                scores.append(0.0)
        return np.array(scores, dtype=np.float32)

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict the hybrid preference score for a user-item pair.

        Args:
            user_id: Target user ID.
            movie_id: Target movie ID.

        Returns:
            float: Weighted hybrid preference score.
        """
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} is not fitted yet.")

        cf_score = 0.0
        try:
            cf_score = float(self.cf_model.predict(user_id, movie_id))
        except (NotImplementedError, AttributeError):
            cf_score = 0.0

        cb_score = 0.0
        try:
            cb_score = float(self.cb_model.predict(user_id, movie_id))
        except (NotImplementedError, AttributeError):
            cb_score = 0.0

        return float(self.alpha * cf_score + (1.0 - self.alpha) * cb_score)

    def recommend(
        self,
        user_id: int,
        top_n: int = 10,
        candidate_items: Optional[Sequence[int]] = None,
    ) -> List[int]:
        """Generate top_n movie recommendations using score-normalized weighted fusion.

        Args:
            user_id: Target user ID.
            top_n: Number of items to recommend (default 10).
            candidate_items: Candidate movie IDs to rank. If None, unseen catalog items.

        Returns:
            List[int]: Top-N recommended movie IDs.
        """
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} is not fitted yet.")

        if candidate_items is None:
            candidate_items = self._get_default_candidates(user_id)

        candidate_list = list(candidate_items)
        if not candidate_list:
            return []

        # Extract raw candidate scores
        cf_raw = self._get_cf_candidate_scores(user_id, candidate_list)
        cb_raw = self._get_cb_candidate_scores(user_id, candidate_list)

        # Normalize component scores across the candidate set
        cf_norm = normalize_vector(cf_raw, method=self.normalization)
        cb_norm = normalize_vector(cb_raw, method=self.normalization)

        # Weighted fusion: alpha * CF + (1 - alpha) * CB
        fused_scores = self.alpha * cf_norm + (1.0 - self.alpha) * cb_norm

        # Sort candidate items descending by fused score, tie-break by candidate movie_id
        ranked_indices = np.lexsort(([-int(m) for m in candidate_list], -fused_scores))
        ranked_candidates = [int(candidate_list[idx]) for idx in ranked_indices]

        return ranked_candidates[:top_n]
