"""Item-Based Collaborative Filtering Recommender (Item-Item KNN).

Predicts a user's preference for an item based on the ratings the user gave to
similar items using Adjusted Cosine similarity with confidence damping.

Formulation:
1. User Centering:
   r_tilde_{u, i} = r_{ui} - mu_u (for observed ratings only)
2. Adjusted Cosine Similarity between items:
   s_{ij} = dot(col_i, col_j) / (||col_i||_2 * ||col_j||_2)
   where col_i is the column vector of user-centered ratings for item i.
   Negative similarities are filtered out (s_{ij} <= 0 -> 0).
3. Nearest Neighbors:
   For each item i, retain top-k most similar items N_k(i).
4. Prediction with Confidence Damping:
   r_hat_{ui} = mu_u + sum_{j in I_u cap N_k(i)} s_{ij} * (r_{uj} - mu_u) / (sum_{j in I_u cap N_k(i)} |s_{ij}| + lambda_damp)
5. Fallback:
   If user u has fewer than min_support items similar to i, falls back to mu_u.
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


class ItemBasedCF(BaseRecommender):
    """Item-Based Collaborative Filtering (Item-Item KNN)."""

    def __init__(
        self,
        name: str = "ItemBasedCF",
        k_neighbors: int = 30,
        min_similarity: float = 0.0,
        damping: float = 1.0,
        min_support: int = 2,
    ) -> None:
        """Initialize ItemBasedCF.

        Args:
            name: Model name.
            k_neighbors: Number of similar items to retain per item (default 30).
            min_similarity: Minimum similarity threshold (default 0.0).
            damping: Confidence damping in prediction denominator (default 1.0).
            min_support: Minimum number of rated similar items required for an item (default 2).
        """
        super().__init__(name=name)
        self.k_neighbors: int = k_neighbors
        self.min_similarity: float = min_similarity
        self.damping: float = damping
        self.min_support: int = min_support

        self.user_means: Dict[int, float] = {}
        self.global_mean: float = 0.0
        self.max_user_id: int = 0
        self.max_movie_id: int = 0

        self.R_centered: Optional[csr_matrix] = None
        self.R_binary: Optional[csr_matrix] = None
        self.S_sparse_T: Optional[csr_matrix] = None
        self.S_abs_T: Optional[csr_matrix] = None

    def fit(self, train_df: pd.DataFrame) -> "ItemBasedCF":
        """Compute item-item similarities strictly from the training dataset.

        Args:
            train_df: Training DataFrame containing ['user_id', 'movie_id', 'rating'].

        Returns:
            self: The fitted model.
        """
        logger.info(
            "Fitting %s (k=%d, damp=%.1f, min_supp=%d) on %d rows...",
            self.name,
            self.k_neighbors,
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

        # 2. Centered ratings: r_{ui} - mu_u
        centered_ratings = train_df["rating"] - train_df["user_id"].map(self.user_means)

        # 3. Sparse user-centered matrix: shape (max_user_id + 1, max_movie_id + 1)
        self.R_centered = csr_matrix(
            (centered_ratings.astype(np.float32), (train_df["user_id"], train_df["movie_id"])),
            shape=(self.max_user_id + 1, self.max_movie_id + 1),
            dtype=np.float32,
        )

        self.R_binary = csr_matrix(
            (np.ones(len(train_df), dtype=np.float32), (train_df["user_id"], train_df["movie_id"])),
            shape=(self.max_user_id + 1, self.max_movie_id + 1),
            dtype=np.float32,
        )

        # 4. Adjusted cosine similarity: normalize columns of user-centered matrix
        # R_item_norm has shape (U+1, M+1), columns are L2 normalized
        R_item_norm = normalize(self.R_centered, norm="l2", axis=0)

        # Item-Item similarity: (M+1, U+1) @ (U+1, M+1) -> (M+1, M+1)
        item_sim = R_item_norm.T.dot(R_item_norm).toarray()
        np.fill_diagonal(item_sim, 0.0)

        # Filter out similarities <= min_similarity
        item_sim[item_sim <= self.min_similarity] = 0.0

        # Retain top-k positive neighbors per item
        top_k_sim = np.zeros_like(item_sim, dtype=np.float32)
        for i in range(self.max_movie_id + 1):
            top_indices = np.argsort(item_sim[i])[::-1][:self.k_neighbors]
            top_k_sim[i, top_indices] = item_sim[i, top_indices]

        # S has shape (M+1, M+1) where S[i, j] is similarity between item i and neighbor j
        S_sparse = csr_matrix(top_k_sim)
        S_abs_sparse = csr_matrix(np.abs(top_k_sim))

        # Store transposed similarity matrices for fast matrix-vector products: shape (M+1, M+1)
        self.S_sparse_T = S_sparse.T.tocsr()
        self.S_abs_T = S_abs_sparse.T.tocsr()

        logger.info("Fitted item similarities across %d items successfully.", len(self.all_train_items))
        return self

    def _score_all_items_for_user(self, user_id: int) -> np.ndarray:
        """Compute predicted scores for all movies for user_id."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        if user_id not in self.all_train_users or user_id > self.max_user_id:
            return np.full(self.max_movie_id + 1, self.global_mean, dtype=np.float32)

        u_mean = self.user_means.get(user_id, self.global_mean)

        # Sparse row for user u: shape (1, M+1)
        r_u_centered = self.R_centered.getrow(user_id)
        r_u_binary = self.R_binary.getrow(user_id)

        # Numerator: (1, M+1) @ (M+1, M+1) -> (1, M+1)
        # Denominator: (1, M+1) @ (M+1, M+1) -> (1, M+1)
        num = r_u_centered.dot(self.S_sparse_T).toarray().ravel()
        denom = r_u_binary.dot(self.S_abs_T).toarray().ravel()
        n_supp = r_u_binary.dot(self.S_sparse_T > 0).toarray().ravel()

        scores = np.full(self.max_movie_id + 1, -999.0, dtype=np.float32)
        valid = (denom > 1e-8) & (n_supp >= self.min_support)
        scores[valid] = u_mean + (num[valid] / (denom[valid] + self.damping))

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
