"""SVD++ (Factorization with Implicit Feedback) Recommender.

This module implements Yehuda Koren's SVD++ algorithm (KDD 2008), extending standard
latent factor models by incorporating implicit feedback derived strictly from the set
of items rated by each user in the training set.

Formulation:
    \\hat{r}_{ui} = \\mu + b_u + b_i + \\mathbf{q}_i^T \\left( \\mathbf{p}_u + |I_u|^{-1/2} \\sum_{j \\in I_u} \\mathbf{y}_j \\right)

Loss:
    \\mathcal{L} = \\frac{1}{2} \\sum_{(u, i) \\in \\mathcal{K}} (r_{ui} - \\hat{r}_{ui})^2
                 + \\frac{\\lambda}{2} \\left( b_u^2 + b_i^2 + \\|\\mathbf{p}_u\\|_2^2 + \\|\\mathbf{q}_i\\|_2^2 + \\sum_{j \\in I_u} \\|\\mathbf{y}_j\\|_2^2 \\right)
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


class SVDppRecommender(BaseRecommender):
    """Yehuda Koren's SVD++ Recommender with Implicit Feedback from Training History."""

    def __init__(
        self,
        n_factors: int = 20,
        learning_rate: float = 0.005,
        regularization: float = 0.02,
        n_epochs: int = 25,
        random_state: int = 42,
        early_stopping: bool = True,
        patience: int = 5,
        name: str = "SVDppRecommender",
    ) -> None:
        """Initialize SVD++ Recommender.

        Args:
            n_factors: Dimension of latent factor space d (default 20).
            learning_rate: Step size gamma for SGD updates (default 0.005).
            regularization: L2 regularization penalty lambda (default 0.02).
            n_epochs: Maximum number of SGD training iterations (default 25).
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
        self.Y: np.ndarray = np.empty((0, 0), dtype=np.float32)

        # Pre-computed implicit feedback vectors per user: |I_u|^(-1/2) * sum(Y[j])
        self.user_impl_vectors: np.ndarray = np.empty((0, 0), dtype=np.float32)

        # Training history logs
        self.train_rmse_history: List[float] = []
        self.val_rmse_history: List[float] = []

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
    ) -> "SVDppRecommender":
        """Fit SVD++ model strictly using training interactions for explicit & implicit data.

        Args:
            train_df: DataFrame with ['user_id', 'movie_id', 'rating'].
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

        # Build user-indexed interaction lists strictly from train_df
        user_items_list: List[List[int]] = [[] for _ in range(n_users)]
        user_ratings_list: List[List[float]] = [[] for _ in range(n_users)]

        for _, row in train_df.iterrows():
            u_idx = self.user_to_idx[int(row["user_id"])]
            i_idx = self.item_to_idx[int(row["movie_id"])]
            r_val = float(row["rating"])
            user_items_list[u_idx].append(i_idx)
            user_ratings_list[u_idx].append(r_val)

        user_items = [np.array(x, dtype=np.int32) for x in user_items_list]
        user_ratings = [np.array(x, dtype=np.float32) for x in user_ratings_list]
        user_sqrt = np.array(
            [1.0 / np.sqrt(len(x)) if len(x) > 0 else 0.0 for x in user_items],
            dtype=np.float32,
        )

        n_samples = len(train_df)
        self.mu = float(train_df["rating"].mean())
        self.bu = np.zeros(n_users, dtype=np.float32)
        self.bi = np.zeros(n_items, dtype=np.float32)

        scale = 1.0 / np.sqrt(self.n_factors)
        self.P = rng.normal(0.0, 0.05 * scale, size=(n_users, self.n_factors)).astype(np.float32)
        self.Q = rng.normal(0.0, 0.05 * scale, size=(n_items, self.n_factors)).astype(np.float32)
        self.Y = rng.normal(0.0, 0.05 * scale, size=(n_items, self.n_factors)).astype(np.float32)

        # Validation tracking setup
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

        user_order = np.arange(n_users)
        lr = self.learning_rate
        reg = self.regularization
        d = self.n_factors

        for epoch in range(1, self.n_epochs + 1):
            rng.shuffle(user_order)
            train_sse = 0.0

            for u in user_order:
                u_its = user_items[u]
                nu = len(u_its)
                if nu == 0:
                    continue

                r_actual = user_ratings[u]
                sqrt_nu = user_sqrt[u]

                # Implicit feedback vector for user u
                u_impl = sqrt_nu * np.sum(self.Y[u_its], axis=0)
                p_eff = self.P[u] + u_impl

                accum_q = np.zeros(d, dtype=np.float32)

                for idx in range(nu):
                    i = u_its[idx]
                    r = r_actual[idx]
                    pred = self.mu + self.bu[u] + self.bi[i] + np.dot(self.Q[i], p_eff)
                    err = r - pred
                    train_sse += float(err * err)

                    self.bu[u] += lr * (err - reg * self.bu[u])
                    self.bi[i] += lr * (err - reg * self.bi[i])

                    q_old = self.Q[i].copy()
                    self.Q[i] += lr * (err * p_eff - reg * q_old)
                    self.P[u] += lr * (err * q_old - reg * self.P[u])
                    accum_q += err * q_old

                # Update implicit factor representations Y for rated items
                grad_y = sqrt_nu * accum_q
                self.Y[u_its] += lr * (grad_y - reg * self.Y[u_its])

            train_rmse = float(np.sqrt(train_sse / n_samples))
            self.train_rmse_history.append(train_rmse)

            # Precompute current user implicit vectors for validation / inference
            self._update_all_user_impl_vectors(user_items, user_sqrt, n_users)

            val_rmse = None
            if val_u is not None and len(val_u) > 0:
                p_eff_val = self.P[val_u] + self.user_impl_vectors[val_u]
                val_preds = self.mu + self.bu[val_u] + self.bi[val_i] + np.sum(
                    p_eff_val * self.Q[val_i], axis=1
                )
                val_rmse = float(np.sqrt(np.mean((val_r - val_preds) ** 2)))
                self.val_rmse_history.append(val_rmse)

                if val_rmse < best_val_rmse - 1e-4:
                    best_val_rmse = val_rmse
                    best_epoch = epoch
                    best_state = (
                        self.bu.copy(),
                        self.bi.copy(),
                        self.P.copy(),
                        self.Q.copy(),
                        self.Y.copy(),
                    )
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
            self.bu, self.bi, self.P, self.Q, self.Y = best_state
            self._update_all_user_impl_vectors(user_items, user_sqrt, n_users)

        self.is_fitted = True
        logger.info(
            "Fitted %s successfully across %d users and %d items. Final Train RMSE=%.4f",
            self.name,
            n_users,
            n_items,
            self.train_rmse_history[-1],
        )
        return self

    def _update_all_user_impl_vectors(
        self,
        user_items: List[np.ndarray],
        user_sqrt: np.ndarray,
        n_users: int,
    ) -> None:
        """Precompute and store implicit feedback vectors |I_u|^(-1/2) * sum(Y[j]) for all users."""
        self.user_impl_vectors = np.zeros((n_users, self.n_factors), dtype=np.float32)
        for u in range(n_users):
            if len(user_items[u]) > 0:
                self.user_impl_vectors[u] = user_sqrt[u] * np.sum(self.Y[user_items[u]], axis=0)

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict the preference rating for a user-item pair using explicit + implicit factors.

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
            p_eff = self.P[u_idx] + self.user_impl_vectors[u_idx]
            score += float(self.bu[u_idx] + self.bi[i_idx] + np.dot(self.Q[i_idx], p_eff))
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
        """Generate top_n movie recommendations for user_id using SVD++ factors.

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
        p_eff = (self.P[u_idx] + self.user_impl_vectors[u_idx]) if u_idx is not None else None

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

            if p_eff is not None:
                dot_scores = np.dot(self.Q[idx_array], p_eff)
                final_scores = base_scores + dot_scores
            else:
                final_scores = base_scores

            for cid, sc in zip(known_cand_ids, final_scores):
                scores_map[cid] = float(sc)

        fallback_score = self.mu + user_bias
        for cid in unknown_cand_ids:
            scores_map[cid] = fallback_score

        ranked = sorted(candidate_list, key=lambda mid: (scores_map[mid], -int(mid)), reverse=True)
        return [int(mid) for mid in ranked[:top_n]]
