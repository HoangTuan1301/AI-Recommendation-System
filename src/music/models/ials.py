"""Implicit Alternating Least Squares (iALS) Recommender.

Implements the classic matrix factorization algorithm for implicit feedback
datasets (Hu, Koren, Volinsky, IEEE ICDM 2008) with logarithmic confidence weighting
and closed-form Gram matrix updates.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import logging
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix

from src.music.models.base_music_recommender import BaseMusicRecommender

logger = logging.getLogger(__name__)


class IALSMusicRecommender(BaseMusicRecommender):
    """Implicit Alternating Least Squares (iALS) Recommender for song-level implicit feedback."""

    def __init__(
        self,
        name: str = "Implicit ALS",
        n_factors: int = 32,
        regularization: float = 0.1,
        alpha: float = 40.0,
        n_iterations: int = 15,
        random_seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            n_factors=n_factors,
            regularization=regularization,
            alpha=alpha,
            n_iterations=n_iterations,
            random_seed=random_seed,
            **kwargs,
        )
        self.n_factors: int = n_factors
        self.regularization: float = regularization
        self.alpha: float = alpha
        self.n_iterations: int = n_iterations
        self.random_seed: int = random_seed

        self.user_factors: Optional[np.ndarray] = None  # (num_users, n_factors)
        self.item_factors: Optional[np.ndarray] = None  # (num_songs, n_factors)

        self.user_to_idx: Dict[str, int] = {}
        self.idx_to_user: Dict[int, str] = {}
        self.song_to_idx: Dict[str, int] = {}
        self.idx_to_song: Dict[int, str] = {}
        self.training_losses: List[float] = []

    def fit(self, train_df: pd.DataFrame) -> "IALSMusicRecommender":
        """Fit latent factor matrices X and Y using Alternating Least Squares."""
        self._init_train_state(train_df)
        start_time = time.time()

        unique_users = sorted(train_df["user_id"].unique())
        unique_songs = self.all_train_items

        self.user_to_idx = {u: i for i, u in enumerate(unique_users)}
        self.idx_to_user = {i: u for i, u in enumerate(unique_users)}
        self.song_to_idx = {s: i for i, s in enumerate(unique_songs)}
        self.idx_to_song = {i: s for i, s in enumerate(unique_songs)}

        num_users = len(unique_users)
        num_songs = len(unique_songs)

        u_indices = np.array([self.user_to_idx[u] for u in train_df["user_id"]], dtype=np.int32)
        s_indices = np.array([self.song_to_idx[s] for s in train_df["song_id"]], dtype=np.int32)
        play_counts = train_df["play_count"].to_numpy(dtype=np.float32)

        # Confidence: c_ui = 1 + alpha * log(1 + r_ui)
        # Weight difference: w_ui = c_ui - 1 = alpha * log(1 + r_ui)
        weights = self.alpha * np.log1p(play_counts)
        confidences = 1.0 + weights

        # Build CSR and CSC matrices of (c_ui - 1)
        C_minus_I_user = csr_matrix((weights, (u_indices, s_indices)), shape=(num_users, num_songs), dtype=np.float32)
        C_minus_I_item = csc_matrix((weights, (u_indices, s_indices)), shape=(num_users, num_songs), dtype=np.float32)

        # Initialize factors with Gaussian noise
        rng = np.random.default_rng(self.random_seed)
        self.user_factors = rng.normal(0, 0.05, size=(num_users, self.n_factors)).astype(np.float32)
        self.item_factors = rng.normal(0, 0.05, size=(num_songs, self.n_factors)).astype(np.float32)

        lambda_eye = self.regularization * np.eye(self.n_factors, dtype=np.float32)

        logger.info(
            f"Starting iALS training: {num_users:,} users, {num_songs:,} songs, "
            f"k={self.n_factors}, iters={self.n_iterations}, lambda={self.regularization}..."
        )

        for iteration in range(1, self.n_iterations + 1):
            iter_start = time.time()

            # 1. Update User Factors
            # Precompute Y^T Y
            YtY = self.item_factors.T @ self.item_factors  # (k, k)

            for u in range(num_users):
                # Get non-zero items for user u
                start_idx = C_minus_I_user.indptr[u]
                end_idx = C_minus_I_user.indptr[u + 1]
                if start_idx == end_idx:
                    # Cold user, regularized zero
                    self.user_factors[u] = np.zeros(self.n_factors, dtype=np.float32)
                    continue

                items = C_minus_I_user.indices[start_idx:end_idx]
                w_u = C_minus_I_user.data[start_idx:end_idx]  # (c_ui - 1)

                Y_u = self.item_factors[items]  # (n_u, k)
                # A = Y^T Y + Y_u^T diag(c_ui - 1) Y_u + lambda * I
                A = YtY + (Y_u.T * w_u) @ Y_u + lambda_eye
                # b = Y_u^T c_ui (since p_ui = 1 for positive train interactions)
                b = Y_u.T @ (1.0 + w_u)

                # Solve A * x_u = b
                try:
                    self.user_factors[u] = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    self.user_factors[u] = np.linalg.lstsq(A, b, rcond=None)[0]

            # 2. Update Item Factors
            # Precompute X^T X
            XtX = self.user_factors.T @ self.user_factors  # (k, k)

            for i in range(num_songs):
                start_idx = C_minus_I_item.indptr[i]
                end_idx = C_minus_I_item.indptr[i + 1]
                if start_idx == end_idx:
                    self.item_factors[i] = np.zeros(self.n_factors, dtype=np.float32)
                    continue

                users = C_minus_I_item.indices[start_idx:end_idx]
                w_i = C_minus_I_item.data[start_idx:end_idx]

                X_i = self.user_factors[users]
                A = XtX + (X_i.T * w_i) @ X_i + lambda_eye
                b = X_i.T @ (1.0 + w_i)

                try:
                    self.item_factors[i] = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    self.item_factors[i] = np.linalg.lstsq(A, b, rcond=None)[0]

            iter_time = time.time() - iter_start
            logger.info(f"iALS Iteration {iteration}/{self.n_iterations} completed in {iter_time:.2f}s.")

        elapsed = time.time() - start_time
        self.is_fitted = True
        logger.info(f"Fitted {self.name} successfully in {elapsed:.2f}s.")
        return self

    def score(self, user_id: str, song_ids: Sequence[str]) -> np.ndarray:
        """Compute inner product scores between user factor and candidate item factors."""
        if not self.is_fitted or self.user_factors is None or self.item_factors is None:
            raise RuntimeError(f"Model '{self.name}' must be fitted before calling score().")

        if user_id not in self.user_to_idx:
            # Global average score for cold user
            return np.zeros(len(song_ids), dtype=np.float32)

        u_idx = self.user_to_idx[user_id]
        u_factor = self.user_factors[u_idx]  # (k,)

        # Extract item indices
        valid_indices = [self.song_to_idx[s] for s in song_ids if s in self.song_to_idx]
        if not valid_indices:
            return np.zeros(len(song_ids), dtype=np.float32)

        scores = np.zeros(len(song_ids), dtype=np.float32)
        for idx, song_id in enumerate(song_ids):
            if song_id in self.song_to_idx:
                s_idx = self.song_to_idx[song_id]
                scores[idx] = float(np.dot(u_factor, self.item_factors[s_idx]))

        return scores
