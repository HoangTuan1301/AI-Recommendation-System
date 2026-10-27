"""Non-Negative Matrix Factorization (NMF) Recommender for Explicit Feedback.

This module implements Non-Negative Matrix Factorization specifically designed
for explicit rating data with missing values. Both user latent factors (W) and item
latent factors (H) are constrained to the non-negative orthant (W >= 0, H >= 0).

Formulation:
    \\hat{r}_{ui} = \\mathbf{w}_u^T \\mathbf{h}_i = \\sum_{k=1}^d W_{uk} H_{ik}

Loss:
    \\mathcal{L} = \\frac{1}{2} \\sum_{(u, i) \\in \\mathcal{K}} (r_{ui} - \\mathbf{w}_u^T \\mathbf{h}_i)^2
                 + \\frac{\\lambda}{2} \\left( \\|\\mathbf{w}_u\\|_2^2 + \\|\\mathbf{h}_i\\|_2^2 \\right)
    \\text{subject to } W_{uk} \\ge 0, \\; H_{ik} \\ge 0

Optimization:
    Projected Stochastic Gradient Descent (Projected SGD):
    After each mini-batch gradient update, factors are projected back onto the non-negative
    orthant via the Euclidean projection operator \\Pi_{\\mathbb{R}_+}(x) = \\max(0, x).
    Crucially, optimization is conducted STRICTLY over observed entries \\mathcal{K};
    unobserved entries are never imputed or treated as zero ratings.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Dict, List, Optional, Sequence, Union
import logging
import numpy as np
import pandas as pd

from src.models.base_recommender import BaseRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NMFRecommender(BaseRecommender):
    """Non-Negative Matrix Factorization Recommender via Projected SGD."""

    def __init__(
        self,
        n_factors: int = 20,
        learning_rate: float = 0.005,
        regularization: float = 0.02,
        n_epochs: int = 30,
        batch_size: int = 512,
        random_state: int = 42,
        early_stopping: bool = True,
        patience: int = 5,
        name: str = "NMFRecommender",
    ) -> None:
        """Initialize NMF Recommender.

        Args:
            n_factors: Number of non-negative latent factors d (default 20).
            learning_rate: Step size gamma for Projected SGD (default 0.005).
            regularization: L2 regularization parameter lambda (default 0.02).
            n_epochs: Maximum number of training epochs (default 30).
            batch_size: Mini-batch size for vectorized updates (default 512).
            random_state: Random seed for deterministic reproducibility (default 42).
            early_stopping: Whether to monitor validation loss and stop early (default True).
            patience: Number of epochs to wait without improvement before stopping (default 5).
            name: Model identifier name.
        """
        super().__init__(name=name)
        self.n_factors = int(n_factors)
        self.learning_rate = float(learning_rate)
        self.regularization = float(regularization)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.random_state = int(random_state)
        self.early_stopping = bool(early_stopping)
        self.patience = int(patience)

        # Internal mappings and parameters
        self.user_to_idx: Dict[int, int] = {}
        self.idx_to_user: Dict[int, int] = {}
        self.item_to_idx: Dict[int, int] = {}
        self.idx_to_item: Dict[int, int] = {}

        self.mu: float = 0.0
        self.user_means: np.ndarray = np.empty(0, dtype=np.float32)
        self.item_means: np.ndarray = np.empty(0, dtype=np.float32)
        self.W: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.H: np.ndarray = np.empty((0, 0), dtype=np.float32)

        # Training history logs
        self.train_rmse_history: List[float] = []
        self.val_rmse_history: List[float] = []

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
    ) -> "NMFRecommender":
        """Fit NMF model strictly on observed training ratings with non-negativity constraints.

        Args:
            train_df: DataFrame containing ['user_id', 'movie_id', 'rating'].
            val_df: Optional validation DataFrame for monitoring and early stopping.

        Returns:
            self: Fitted recommender instance.
        """
        self._store_train_metadata(train_df)
        rng = np.random.RandomState(self.random_state)

        unique_users = sorted(train_df["user_id"].unique())
        unique_items = sorted(train_df["movie_id"].unique())
        self.user_to_idx = {u: idx for idx, u in enumerate(unique_users)}
        self.idx_to_user = {idx: u for idx, u in enumerate(unique_users)}
        self.item_to_idx = {i: idx for idx, i in enumerate(unique_items)}
        self.idx_to_item = {idx: i for idx, i in enumerate(unique_items)}

        n_users = len(unique_users)
        n_items = len(unique_items)

        u_arr = train_df["user_id"].map(self.user_to_idx).values.astype(np.int32)
        i_arr = train_df["movie_id"].map(self.item_to_idx).values.astype(np.int32)
        r_arr = train_df["rating"].values.astype(np.float32)
        n_samples = len(r_arr)

        self.mu = float(np.mean(r_arr))

        # Compute empirical user and item averages for fallback
        self.user_means = np.full(n_users, self.mu, dtype=np.float32)
        self.item_means = np.full(n_items, self.mu, dtype=np.float32)
        user_grouped = train_df.groupby("user_id")["rating"].mean()
        item_grouped = train_df.groupby("movie_id")["rating"].mean()
        for u, val in user_grouped.items():
            self.user_means[self.user_to_idx[u]] = float(val)
        for i, val in item_grouped.items():
            self.item_means[self.item_to_idx[i]] = float(val)

        # Initialize W and H with strictly positive uniform distribution
        # Scale chosen such that average dot product W[u] . H[i] ~ mu
        init_val = np.sqrt(self.mu / self.n_factors)
        self.W = rng.uniform(init_val * 0.7, init_val * 1.3, size=(n_users, self.n_factors)).astype(np.float32)
        self.H = rng.uniform(init_val * 0.7, init_val * 1.3, size=(n_items, self.n_factors)).astype(np.float32)

        # Validation data preparation
        val_u, val_i, val_r = None, None, None
        if val_df is not None:
            valid_val = val_df[
                val_df["user_id"].isin(self.user_to_idx) & val_df["movie_id"].isin(self.item_to_idx)
            ]
            val_u = valid_val["user_id"].map(self.user_to_idx).values.astype(np.int32)
            val_i = valid_val["movie_id"].map(self.item_to_idx).values.astype(np.int32)
            val_r = valid_val["rating"].values.astype(np.float32)

        self.train_rmse_history = []
        self.val_rmse_history = []

        best_val_rmse = float("inf")
        best_epoch = -1
        best_state = None
        no_improve_count = 0

        logger.info(
            "Fitting %s (factors=%d, lr=%.4f, reg=%.4f, epochs=%d, seed=%d) on %d rows...",
            self.name,
            self.n_factors,
            self.learning_rate,
            self.regularization,
            self.n_epochs,
            self.random_state,
            n_samples,
        )

        indices = np.arange(n_samples)
        lr = self.learning_rate
        reg = self.regularization

        for epoch in range(1, self.n_epochs + 1):
            rng.shuffle(indices)
            train_sse = 0.0

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch = indices[start:end]
                u_batch = u_arr[batch]
                i_batch = i_arr[batch]
                r_batch = r_arr[batch]

                preds = np.sum(self.W[u_batch] * self.H[i_batch], axis=1)
                errs = r_batch - preds
                train_sse += float(np.sum(errs ** 2))

                w_old = self.W[u_batch].copy()
                h_old = self.H[i_batch].copy()

                np.add.at(self.W, u_batch, lr * (errs[:, None] * h_old - reg * w_old))
                np.add.at(self.H, i_batch, lr * (errs[:, None] * w_old - reg * h_old))

                # Non-negative projection on updated users and items
                unique_u = np.unique(u_batch)
                unique_i = np.unique(i_batch)
                self.W[unique_u] = np.maximum(0.0, self.W[unique_u])
                self.H[unique_i] = np.maximum(0.0, self.H[unique_i])

            train_rmse = float(np.sqrt(train_sse / n_samples))
            self.train_rmse_history.append(train_rmse)

            # Validation tracking
            val_rmse = None
            if val_u is not None and len(val_u) > 0:
                val_preds = np.sum(self.W[val_u] * self.H[val_i], axis=1)
                val_rmse = float(np.sqrt(np.mean((val_r - val_preds) ** 2)))
                self.val_rmse_history.append(val_rmse)

                if val_rmse < best_val_rmse - 1e-4:
                    best_val_rmse = val_rmse
                    best_epoch = epoch
                    best_state = (self.W.copy(), self.H.copy())
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if self.early_stopping and no_improve_count >= self.patience:
                    logger.info(
                        "Early stopping triggered at epoch %d. Best Val RMSE=%.4f at epoch %d.",
                        epoch,
                        best_val_rmse,
                        best_epoch,
                    )
                    break

        if self.early_stopping and best_state is not None:
            self.W, self.H = best_state

        # Assert non-negativity guarantee
        assert np.all(self.W >= 0.0), "Critical Error: W violates non-negativity!"
        assert np.all(self.H >= 0.0), "Critical Error: H violates non-negativity!"

        self.is_fitted = True
        logger.info(
            "Fitted %s successfully across %d users and %d items. Final Train RMSE=%.4f (W min=%.4f, H min=%.4f)",
            self.name,
            n_users,
            n_items,
            self.train_rmse_history[-1],
            float(np.min(self.W)),
            float(np.min(self.H)),
        )
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict preference rating using non-negative latent dot product.

        Args:
            user_id: Target user ID.
            movie_id: Target movie ID.

        Returns:
            float: Predicted non-negative rating score.
        """
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} is not fitted yet.")

        u_idx = self.user_to_idx.get(user_id)
        i_idx = self.item_to_idx.get(movie_id)

        if u_idx is not None and i_idx is not None:
            return float(np.dot(self.W[u_idx], self.H[i_idx]))
        elif u_idx is not None:
            return float(self.user_means[u_idx])
        elif i_idx is not None:
            return float(self.item_means[i_idx])
        return self.mu

    def recommend(
        self,
        user_id: int,
        top_n: int = 10,
        candidate_items: Optional[Sequence[int]] = None,
    ) -> List[int]:
        """Generate top_n movie recommendations for user_id using NMF factors.

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

        u_idx = self.user_to_idx.get(user_id)
        user_factor = self.W[u_idx] if u_idx is not None else None

        known_indices = []
        known_cand_ids = []
        unknown_cand_ids = []

        for cid in candidate_list:
            idx = self.item_to_idx.get(cid)
            if idx is not None:
                known_indices.append(idx)
                known_cand_ids.append(cid)
            else:
                unknown_cand_ids.append(cid)

        scores_map: Dict[int, float] = {}

        if known_indices:
            idx_array = np.array(known_indices, dtype=np.int32)
            if user_factor is not None:
                dot_scores = np.dot(self.H[idx_array], user_factor)
            else:
                dot_scores = self.item_means[idx_array]

            for cid, sc in zip(known_cand_ids, dot_scores):
                scores_map[cid] = float(sc)

        fallback_score = float(self.user_means[u_idx]) if u_idx is not None else self.mu
        for cid in unknown_cand_ids:
            scores_map[cid] = fallback_score

        ranked = sorted(candidate_list, key=lambda mid: (scores_map[mid], -int(mid)), reverse=True)
        return [int(mid) for mid in ranked[:top_n]]
