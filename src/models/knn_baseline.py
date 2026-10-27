"""KNN with Baseline Recommender (KNNBaseline).

An advanced neighborhood model that combines baseline predictors (user and item biases)
with k-nearest neighbors on the residual ratings (Koren, 2010), enhanced with confidence
damping and minimum neighbor support.

Key formulation:
1. Regularized Baseline Predictor:
   b_{ui} = mu + b_u + b_i
   where:
       mu: global mean rating
       b_i = sum_{u in U_i} (r_{ui} - mu) / (lambda_item + |U_i|)
       b_u = sum_{i in I_u} (r_{ui} - mu - b_i) / (lambda_user + |I_u|)
2. Residuals:
   r_tilde_{ui} = r_{ui} - b_{ui}
3. Shrunk Pearson Similarity on Residuals:
   s_{ij} = [dot(col_i, col_j) / (||col_i||_2 * ||col_j||_2)] * [n_{ij} / (n_{ij} + lambda_shrink)]
   where n_{ij} = |U_i cap U_j| is the number of co-raters.
4. Prediction with Confidence Damping:
   r_hat_{ui} = b_{ui} + sum_{j in I_u cap N_k(i)} s_{ij} * (r_{uj} - b_{uj}) / (sum_{j in I_u cap N_k(i)} |s_{ij}| + lambda_damp)
   If fewer than min_support similar items exist (or denominator is 0), falls back to b_{ui} = mu + b_u + b_i.
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


class KNNBaseline(BaseRecommender):
    """KNN with Baseline regularization for Collaborative Filtering."""

    def __init__(
        self,
        name: str = "KNNBaseline",
        k_neighbors: int = 30,
        lambda_user: float = 10.0,
        lambda_item: float = 25.0,
        lambda_shrink: float = 100.0,
        min_similarity: float = 0.0,
        damping: float = 1.0,
        min_support: int = 2,
    ) -> None:
        """Initialize KNNBaseline.

        Args:
            name: Model name.
            k_neighbors: Number of nearest neighbors (default 30).
            lambda_user: L2 regularization for user bias (default 10.0).
            lambda_item: L2 regularization for item bias (default 25.0).
            lambda_shrink: Shrinkage parameter for similarity weighting (default 100.0).
            min_similarity: Minimum positive similarity threshold (default 0.0).
            damping: Confidence damping in prediction denominator (default 1.0).
            min_support: Minimum number of rated similar items required (default 2).
        """
        super().__init__(name=name)
        self.k_neighbors: int = k_neighbors
        self.lambda_user: float = lambda_user
        self.lambda_item: float = lambda_item
        self.lambda_shrink: float = lambda_shrink
        self.min_similarity: float = min_similarity
        self.damping: float = damping
        self.min_support: int = min_support

        self.global_mean: float = 0.0
        self.b_u: Dict[int, float] = {}
        self.b_i: Dict[int, float] = {}
        self.max_user_id: int = 0
        self.max_movie_id: int = 0

        self.R_res: Optional[csr_matrix] = None
        self.R_bin: Optional[csr_matrix] = None
        self.S_sparse_T: Optional[csr_matrix] = None
        self.S_abs_T: Optional[csr_matrix] = None

    def fit(self, train_df: pd.DataFrame) -> "KNNBaseline":
        """Fit baseline predictors and compute residual item similarities strictly on Train.

        Args:
            train_df: Training DataFrame containing ['user_id', 'movie_id', 'rating'].

        Returns:
            self: The fitted model.
        """
        logger.info(
            "Fitting %s (k=%d, lam_u=%.1f, lam_i=%.1f, lam_s=%.1f, damp=%.1f, min_supp=%d) on %d rows...",
            self.name,
            self.k_neighbors,
            self.lambda_user,
            self.lambda_item,
            self.lambda_shrink,
            self.damping,
            self.min_support,
            len(train_df),
        )
        self._store_train_metadata(train_df)

        self.global_mean = float(train_df["rating"].mean())
        self.max_user_id = int(train_df["user_id"].max())
        self.max_movie_id = int(train_df["movie_id"].max())

        mu = self.global_mean

        # 1. Regularized Item Bias b_i
        item_sums = train_df.groupby("movie_id")["rating"].apply(lambda s: (s - mu).sum()).to_dict()
        item_counts = train_df.groupby("movie_id").size().to_dict()
        self.b_i = {
            mid: float(item_sums.get(mid, 0.0) / (self.lambda_item + item_counts.get(mid, 0)))
            for mid in range(self.max_movie_id + 1)
        }

        # 2. Regularized User Bias b_u
        df = train_df.copy()
        df["b_i"] = df["movie_id"].map(self.b_i).fillna(0.0)
        user_sums = df.groupby("user_id").apply(lambda g: (g["rating"] - mu - g["b_i"]).sum()).to_dict()
        user_counts = df.groupby("user_id").size().to_dict()
        self.b_u = {
            uid: float(user_sums.get(uid, 0.0) / (self.lambda_user + user_counts.get(uid, 0)))
            for uid in range(self.max_user_id + 1)
        }

        # 3. Residual ratings: r_tilde_{ui} = r_{ui} - (mu + b_u + b_i)
        df["b_u"] = df["user_id"].map(self.b_u).fillna(0.0)
        df["residual"] = (df["rating"] - (mu + df["b_u"] + df["b_i"])).astype(np.float32)

        # 4. Sparse residual matrix: shape (max_user_id + 1, max_movie_id + 1)
        self.R_res = csr_matrix(
            (df["residual"], (df["user_id"], df["movie_id"])),
            shape=(self.max_user_id + 1, self.max_movie_id + 1),
            dtype=np.float32,
        )

        self.R_bin = csr_matrix(
            (np.ones(len(df), dtype=np.float32), (df["user_id"], df["movie_id"])),
            shape=(self.max_user_id + 1, self.max_movie_id + 1),
            dtype=np.float32,
        )

        # 5. Co-rated support for shrinkage: n_{ij} = |U_i cap U_j|
        co_rated = self.R_bin.T.dot(self.R_bin).toarray()

        # 6. Cosine similarity on residuals (column normalized)
        R_res_norm = normalize(self.R_res, norm="l2", axis=0)
        raw_sim = R_res_norm.T.dot(R_res_norm).toarray()
        np.fill_diagonal(raw_sim, 0.0)

        # 7. Apply shrinkage and filter non-positive
        shrunk_sim = raw_sim * (co_rated / (co_rated + self.lambda_shrink))
        shrunk_sim[shrunk_sim <= self.min_similarity] = 0.0

        # Retain top-k positive neighbors per item
        top_k_sim = np.zeros_like(shrunk_sim, dtype=np.float32)
        for i in range(self.max_movie_id + 1):
            top_indices = np.argsort(shrunk_sim[i])[::-1][:self.k_neighbors]
            top_k_sim[i, top_indices] = shrunk_sim[i, top_indices]

        S_sparse = csr_matrix(top_k_sim)
        S_abs = csr_matrix(np.abs(top_k_sim))

        self.S_sparse_T = S_sparse.T.tocsr()
        self.S_abs_T = S_abs.T.tocsr()

        logger.info("Fitted KNNBaseline across %d users and %d items.", len(self.all_train_users), len(self.all_train_items))
        return self

    def _score_all_items_for_user(self, user_id: int) -> np.ndarray:
        """Compute predicted scores for all movies for user_id."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        u_bias = self.b_u.get(user_id, 0.0)
        mu = self.global_mean

        # Baseline vector for this user across all items: mu + b_u + b_i
        b_i_arr = np.array([self.b_i.get(i, 0.0) for i in range(self.max_movie_id + 1)], dtype=np.float32)
        baseline = (mu + u_bias + b_i_arr).astype(np.float32)

        if user_id not in self.all_train_users or user_id > self.max_user_id:
            return baseline

        # Sparse row for user u: shape (1, M+1)
        r_u_res = self.R_res.getrow(user_id)
        r_u_bin = self.R_bin.getrow(user_id)

        # Numerator: r_u_res @ S_sparse_T -> (1, M+1)
        # Denominator: r_u_bin @ S_abs_T -> (1, M+1)
        num = r_u_res.dot(self.S_sparse_T).toarray().ravel()
        denom = r_u_bin.dot(self.S_abs_T).toarray().ravel()
        n_supp = r_u_bin.dot(self.S_sparse_T > 0).toarray().ravel()

        scores = baseline.copy()
        valid = (denom > 1e-8) & (n_supp >= self.min_support)
        scores[valid] += (num[valid] / (denom[valid] + self.damping))
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
            return self.global_mean + self.b_u.get(user_id, 0.0)

        all_scores = self._score_all_items_for_user(user_id)
        val = float(all_scores[movie_id])
        if np.isnan(val) or np.isinf(val):
            return self.global_mean + self.b_u.get(user_id, 0.0) + self.b_i.get(movie_id, 0.0)
        return val
