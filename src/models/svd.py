"""Funk SVD (Regularized Matrix Factorization) Recommender.

This module implements the seminal Simon Funk SVD algorithm with user/item biases
and L2 regularization, optimized using mini-batch Stochastic Gradient Descent (SGD)
strictly over observed explicit ratings.

Formulation:
    \\hat{r}_{ui} = \\mu + b_u + b_i + \\mathbf{p}_u^T \\mathbf{q}_i

Loss:
    \\mathcal{L} = \\frac{1}{2} \\sum_{(u, i) \\in \\mathcal{K}} (r_{ui} - \\hat{r}_{ui})^2
                 + \\frac{\\lambda}{2} \\left( b_u^2 + b_i^2 + \\|\\mathbf{p}_u\\|_2^2 + \\|\\mathbf{q}_i\\|_2^2 \\right)
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


class SVDRecommender(BaseRecommender):
    """Simon Funk SVD Recommender with User/Item Biases and L2 Regularization."""

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
        name: str = "SVDRecommender",
    ) -> None:
        """Initialize the Funk SVD Recommender.

        Args:
            n_factors: Number of latent dimensions d (default 20).
            learning_rate: Step size gamma for SGD updates (default 0.005).
            regularization: L2 regularization penalty lambda (default 0.02).
            n_epochs: Maximum number of SGD training iterations (default 30).
            batch_size: Mini-batch size for vectorized SGD (default 512).
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
        self.bu: np.ndarray = np.empty(0, dtype=np.float32)
        self.bi: np.ndarray = np.empty(0, dtype=np.float32)
        self.P: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.Q: np.ndarray = np.empty((0, 0), dtype=np.float32)

        # Training history logs
        self.train_rmse_history: List[float] = []
        self.val_rmse_history: List[float] = []

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
    ) -> "SVDRecommender":
        """Fit the Funk SVD model strictly on training interactions.

        Args:
            train_df: DataFrame with ['user_id', 'movie_id', 'rating'].
            val_df: Optional validation DataFrame for monitoring and early stopping.

        Returns:
            self: Fitted recommender instance.
        """
        self._store_train_metadata(train_df)
        rng = np.random.RandomState(self.random_state)

        # Build contiguous 0-based index mappings
        unique_users = sorted(train_df["user_id"].unique())
        unique_items = sorted(train_df["movie_id"].unique())
        self.user_to_idx = {u: idx for idx, u in enumerate(unique_users)}
        self.idx_to_user = {idx: u for idx, u in enumerate(unique_users)}
        self.item_to_idx = {i: idx for idx, i in enumerate(unique_items)}
        self.idx_to_item = {idx: i for idx, i in enumerate(unique_items)}

        n_users = len(unique_users)
        n_items = len(unique_items)

        # Extract training arrays
        u_arr = train_df["user_id"].map(self.user_to_idx).values.astype(np.int32)
        i_arr = train_df["movie_id"].map(self.item_to_idx).values.astype(np.int32)
        r_arr = train_df["rating"].values.astype(np.float32)
        n_samples = len(r_arr)

        # Initialize global mean, biases, and latent factors
        self.mu = float(np.mean(r_arr))
        self.bu = np.zeros(n_users, dtype=np.float32)
        self.bi = np.zeros(n_items, dtype=np.float32)

        scale = 1.0 / np.sqrt(self.n_factors)
        self.P = rng.normal(0.0, 0.05 * scale, size=(n_users, self.n_factors)).astype(np.float32)
        self.Q = rng.normal(0.0, 0.05 * scale, size=(n_items, self.n_factors)).astype(np.float32)

        # Prepare validation data if provided
        val_u, val_i, val_r = None, None, None
        if val_df is not None:
            # Keep only validation entries with seen users and items
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

                preds = self.mu + self.bu[u_batch] + self.bi[i_batch] + np.sum(
                    self.P[u_batch] * self.Q[i_batch], axis=1
                )
                errs = r_batch - preds
                train_sse += float(np.sum(errs ** 2))

                p_old = self.P[u_batch].copy()
                q_old = self.Q[i_batch].copy()

                np.add.at(self.bu, u_batch, lr * (errs - reg * self.bu[u_batch]))
                np.add.at(self.bi, i_batch, lr * (errs - reg * self.bi[i_batch]))
                np.add.at(self.P, u_batch, lr * (errs[:, None] * q_old - reg * p_old))
                np.add.at(self.Q, i_batch, lr * (errs[:, None] * p_old - reg * q_old))

            train_rmse = float(np.sqrt(train_sse / n_samples))
            self.train_rmse_history.append(train_rmse)

            # Validation tracking
            val_rmse = None
            if val_u is not None and len(val_u) > 0:
                val_preds = self.mu + self.bu[val_u] + self.bi[val_i] + np.sum(
                    self.P[val_u] * self.Q[val_i], axis=1
                )
                val_rmse = float(np.sqrt(np.mean((val_r - val_preds) ** 2)))
                self.val_rmse_history.append(val_rmse)

                if val_rmse < best_val_rmse - 1e-4:
                    best_val_rmse = val_rmse
                    best_epoch = epoch
                    best_state = (self.bu.copy(), self.bi.copy(), self.P.copy(), self.Q.copy())
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

        # Restore best checkpoint if early stopping was active
        if self.early_stopping and best_state is not None:
            self.bu, self.bi, self.P, self.Q = best_state

        self.is_fitted = True
        logger.info(
            "Fitted %s successfully across %d users and %d items. Final Train RMSE=%.4f",
            self.name,
            n_users,
            n_items,
            self.train_rmse_history[-1],
        )
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict the preference rating for a user-item pair.

        Args:
            user_id: Target user ID.
            movie_id: Target movie ID.

        Returns:
            float: Predicted rating.
        """
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} is not fitted yet.")

        score = self.mu
        u_idx = self.user_to_idx.get(user_id)
        i_idx = self.item_to_idx.get(movie_id)

        if u_idx is not None and i_idx is not None:
            score += float(self.bu[u_idx] + self.bi[i_idx] + np.dot(self.P[u_idx], self.Q[i_idx]))
        elif u_idx is not None:
            score += float(self.bu[u_idx])
        elif i_idx is not None:
            score += float(self.bi[i_idx])

        return score

    def recommend(
        self,
        user_id: int,
        top_n: int = 10,
        candidate_items: Optional[Sequence[int]] = None,
    ) -> List[int]:
        """Generate top_n movie recommendations for user_id from candidates.

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
        user_bias = float(self.bu[u_idx]) if u_idx is not None else 0.0
        user_factor = self.P[u_idx] if u_idx is not None else None

        # Collect valid candidates with known item indices
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
            item_biases = self.bi[idx_array]
            base_scores = self.mu + user_bias + item_biases

            if user_factor is not None:
                dot_scores = np.dot(self.Q[idx_array], user_factor)
                final_scores = base_scores + dot_scores
            else:
                final_scores = base_scores

            for cid, sc in zip(known_cand_ids, final_scores):
                scores_map[cid] = float(sc)

        # For completely unknown items, score is global baseline
        fallback_score = self.mu + user_bias
        for cid in unknown_cand_ids:
            scores_map[cid] = fallback_score

        # Sort descending by score, tie-break by candidate movie_id
        ranked = sorted(candidate_list, key=lambda mid: (scores_map[mid], -int(mid)), reverse=True)
        return [int(mid) for mid in ranked[:top_n]]
