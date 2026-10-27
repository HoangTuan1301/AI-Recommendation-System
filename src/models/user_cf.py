"""User-Based Collaborative Filtering Recommender (User-User KNN).

Predicts user preference based on the ratings of the most similar users (neighbors)
using Pearson correlation with significance weighting and confidence damping to prevent
single-neighbor obscure item inflation (Herlocker et al., 1999).

Formulation:
1. User Centering:
   r_tilde_{u, i} = r_{ui} - mu_u (for observed ratings only)
2. Pearson Correlation with Significance Weighting:
   s_{uv} = dot(r_tilde_u, r_tilde_v) / (||r_tilde_u||_2 * ||r_tilde_v||_2)
   s'_{uv} = s_{uv} * min(|I_u cap I_v| / gamma, 1.0)  (default gamma=5.0)
3. Prediction with Confidence Damping & Minimum Support:
   r_hat_{ui} = mu_u + sum_{v in N_k(u)} s'_{uv} * (r_{vi} - mu_v) / (sum_{v in N_k(u)} |s'_{uv}| + lambda_damp)
   where N_k(u) are top-k neighbors with s'_{uv} > 0.
4. Fallback:
   If fewer than min_support neighbors rated item i, or denominator is near 0, falls back to mu_u.
"""

from pathlib import Path
import sys

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Dict, List, Optional, Sequence, Set
import logging
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from src.models.base_recommender import BaseRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class UserBasedCF(BaseRecommender):
    """User-Based Collaborative Filtering (User-User KNN)."""

    def __init__(
        self,
        name: str = "UserBasedCF",
        k_neighbors: int = 40,
        min_similarity: float = 0.0,
        significance_gamma: float = 5.0,
        damping: float = 1.0,
        min_support: int = 2,
    ) -> None:
        """Initialize UserBasedCF.

        Args:
            name: Model name.
            k_neighbors: Number of nearest neighbors to consider (default 40).
            min_similarity: Minimum similarity threshold for neighbors (default 0.0).
            significance_gamma: Threshold for significance weighting on co-rated items (default 5.0).
            damping: Confidence damping in prediction denominator (default 1.0).
            min_support: Minimum number of neighbors who rated an item for prediction (default 2).
        """
        super().__init__(name=name)
        self.k_neighbors: int = k_neighbors
        self.min_similarity: float = min_similarity
        self.significance_gamma: float = significance_gamma
        self.damping: float = damping
        self.min_support: int = min_support

        self.user_means: Dict[int, float] = {}
        self.global_mean: float = 0.0
        self.max_user_id: int = 0
        self.max_movie_id: int = 0

        self.user_sim_matrix: Optional[np.ndarray] = None
        self.R_centered_T: Optional[csr_matrix] = None
        self.R_binary_T: Optional[csr_matrix] = None

    def fit(self, train_df: pd.DataFrame) -> "UserBasedCF":
        """Compute user similarities strictly from the training dataset.

        Args:
            train_df: Training DataFrame containing ['user_id', 'movie_id', 'rating'].

        Returns:
            self: The fitted model.
        """
        logger.info(
            "Fitting %s (k=%d, gamma=%.1f, damp=%.1f, min_supp=%d) on %d rows...",
            self.name,
            self.k_neighbors,
            self.significance_gamma,
            self.damping,
            self.min_support,
            len(train_df),
        )
        self._store_train_metadata(train_df)

        self.global_mean = float(train_df["rating"].mean())
        self.max_user_id = int(train_df["user_id"].max())
        self.max_movie_id = int(train_df["movie_id"].max())

        # 1. Compute user mean ratings
        user_means_df = train_df.groupby("user_id")["rating"].mean()
        self.user_means = user_means_df.to_dict()

        # 2. Centered ratings: r_{ui} - mu_u (unobserved ratings are NOT zero ratings, they are absent)
        centered_ratings = train_df["rating"] - train_df["user_id"].map(self.user_means)

        # 3. Build sparse centered interaction matrix: shape (max_user_id + 1, max_movie_id + 1)
        R_centered = csr_matrix(
            (centered_ratings.astype(np.float32), (train_df["user_id"], train_df["movie_id"])),
            shape=(self.max_user_id + 1, self.max_movie_id + 1),
            dtype=np.float32,
        )

        # Binary indicator matrix for support counting: shape (max_user_id + 1, max_movie_id + 1)
        R_binary = csr_matrix(
            (np.ones(len(train_df), dtype=np.float32), (train_df["user_id"], train_df["movie_id"])),
            shape=(self.max_user_id + 1, self.max_movie_id + 1),
            dtype=np.float32,
        )

        # Store transposed matrices for high-speed SpMV column evaluation: shape (M+1, U+1)
        self.R_centered_T = R_centered.T.tocsr()
        self.R_binary_T = R_binary.T.tocsr()

        # 4. Co-rated items between users: (U+1, M+1) @ (M+1, U+1) -> (U+1, U+1)
        co_rated = R_binary.dot(R_binary.T).toarray()

        # 5. Pearson correlation between users via L2-normalized centered vectors
        R_norm = normalize(R_centered, norm="l2", axis=1)
        raw_sim = R_norm.dot(R_norm.T).toarray()
        np.fill_diagonal(raw_sim, 0.0)

        # 6. Apply significance weighting (Herlocker et al. 1999) to quench spurious low-support similarities
        if self.significance_gamma > 0.0:
            sig_weights = np.minimum(co_rated / self.significance_gamma, 1.0)
            self.user_sim_matrix = raw_sim * sig_weights
        else:
            self.user_sim_matrix = raw_sim

        self.user_sim_matrix[self.user_sim_matrix <= self.min_similarity] = 0.0

        logger.info("Fitted %d users across %d items successfully.", len(self.all_train_users), len(self.all_train_items))
        return self

    def _score_all_items_for_user(self, user_id: int) -> np.ndarray:
        """Compute predicted scores for all movies for user_id."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        if user_id not in self.all_train_users or user_id > self.max_user_id:
            return np.full(self.max_movie_id + 1, self.global_mean, dtype=np.float32)

        u_mean = self.user_means.get(user_id, self.global_mean)
        sims = self.user_sim_matrix[user_id]

        # Select top-k positive neighbors
        sorted_neighbors = np.argsort(sims)[::-1]
        top_k_neighbors = [
            v for v in sorted_neighbors[:self.k_neighbors]
            if sims[v] > self.min_similarity
        ]

        if not top_k_neighbors:
            return np.full(self.max_movie_id + 1, u_mean, dtype=np.float32)

        # Build sparse weight vector for neighbors: shape (U+1,)
        sim_weights = np.zeros(self.max_user_id + 1, dtype=np.float32)
        sim_weights[top_k_neighbors] = sims[top_k_neighbors]

        # Binary indicator for counting neighbors who rated each item
        bin_weights = np.zeros(self.max_user_id + 1, dtype=np.float32)
        bin_weights[top_k_neighbors] = 1.0

        # SpMV: (M+1, U+1) @ (U+1,) -> (M+1,)
        num = self.R_centered_T.dot(sim_weights)
        denom = self.R_binary_T.dot(np.abs(sim_weights))
        neighbor_support = self.R_binary_T.dot(bin_weights)

        scores = np.full(self.max_movie_id + 1, -999.0, dtype=np.float32)
        valid = (denom > 1e-8) & (neighbor_support >= self.min_support)
        scores[valid] = u_mean + (num[valid] / (denom[valid] + self.damping))

        # Items without sufficient neighbor support fallback to user mean
        scores[~valid] = u_mean
        return scores

    def recommend(
        self,
        user_id: int,
        top_n: int = 10,
        candidate_items: Optional[Sequence[int]] = None,
    ) -> List[int]:
        """Generate top-N recommendations for user_id.

        Args:
            user_id: Target user ID.
            top_n: Number of items to recommend.
            candidate_items: Candidate movie IDs. If None, uses all unseen items.

        Returns:
            List[int]: Top-N ranked unique movie IDs.
        """
        user_seen = self.user_train_history.get(user_id, set())

        if candidate_items is not None:
            candidates = [
                mid for mid in candidate_items
                if mid not in user_seen and 0 <= mid <= self.max_movie_id
            ]
        else:
            candidates = [
                mid for mid in self.all_train_items
                if mid not in user_seen and 0 <= mid <= self.max_movie_id
            ]

        if not candidates:
            return []

        all_scores = self._score_all_items_for_user(user_id)
        cand_indices = np.array(candidates, dtype=int)
        cand_scores = all_scores[cand_indices]

        scored_pairs = list(zip(candidates, cand_scores))
        scored_pairs.sort(key=lambda x: (-x[1], x[0]))

        return [mid for mid, _ in scored_pairs[:top_n]]

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict rating for user_id on movie_id.

        Args:
            user_id: User ID.
            movie_id: Movie ID.

        Returns:
            float: Predicted rating.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        if movie_id < 0 or movie_id > self.max_movie_id:
            return self.global_mean

        all_scores = self._score_all_items_for_user(user_id)
        val = float(all_scores[movie_id])
        if np.isnan(val) or np.isinf(val):
            return self.global_mean
        return val
