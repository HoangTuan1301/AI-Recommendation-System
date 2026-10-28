"""Bayesian Personalized Ranking Matrix Factorization (BPR-MF) Recommender.

Optimizes pairwise ranking criterion (Rendle et al., UAI 2009) via stochastic
gradient descent with mini-batch negative sampling over unobserved songs.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import logging
import time
import numpy as np
import pandas as pd

from src.music.models.base_music_recommender import BaseMusicRecommender

logger = logging.getLogger(__name__)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


class BPRMFMusicRecommender(BaseMusicRecommender):
    """Bayesian Personalized Ranking Matrix Factorization with Pairwise Ranking Loss."""

    def __init__(
        self,
        name: str = "BPR-MF",
        n_factors: int = 32,
        learning_rate: float = 0.05,
        regularization: float = 0.01,
        n_epochs: int = 20,
        batch_size: int = 1024,
        random_seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            n_factors=n_factors,
            learning_rate=learning_rate,
            regularization=regularization,
            n_epochs=n_epochs,
            batch_size=batch_size,
            random_seed=random_seed,
            **kwargs,
        )
        self.n_factors: int = n_factors
        self.learning_rate: float = learning_rate
        self.regularization: float = regularization
        self.n_epochs: int = n_epochs
        self.batch_size: int = batch_size
        self.random_seed: int = random_seed

        self.user_factors: Optional[np.ndarray] = None  # (num_users, n_factors)
        self.item_factors: Optional[np.ndarray] = None  # (num_songs, n_factors)

        self.user_to_idx: Dict[str, int] = {}
        self.idx_to_user: Dict[int, str] = {}
        self.song_to_idx: Dict[str, int] = {}
        self.idx_to_song: Dict[int, str] = {}
        self.training_losses: List[float] = []

    def fit(self, train_df: pd.DataFrame) -> "BPRMFMusicRecommender":
        """Fit latent factor embeddings using mini-batch BPR-SGD."""
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

        # Build user positive items sets for fast negative sampling rejection
        user_positives: Dict[int, Set[int]] = {u: set() for u in range(num_users)}
        u_arr = np.array([self.user_to_idx[u] for u in train_df["user_id"]], dtype=np.int32)
        i_arr = np.array([self.song_to_idx[s] for s in train_df["song_id"]], dtype=np.int32)

        for u, i in zip(u_arr, i_arr):
            user_positives[u].add(i)

        n_interactions = len(train_df)

        # Initialize factors with Gaussian normal distribution
        rng = np.random.default_rng(self.random_seed)
        self.user_factors = rng.normal(0, 0.05, size=(num_users, self.n_factors)).astype(np.float32)
        self.item_factors = rng.normal(0, 0.05, size=(num_songs, self.n_factors)).astype(np.float32)

        logger.info(
            f"Starting BPR-MF training: {num_users:,} users, {num_songs:,} songs, "
            f"k={self.n_factors}, epochs={self.n_epochs}, lr={self.learning_rate}..."
        )

        for epoch in range(1, self.n_epochs + 1):
            epoch_start = time.time()
            epoch_loss = 0.0

            # Shuffle positive interaction order
            shuffle_idx = rng.permutation(n_interactions)
            shuffled_u = u_arr[shuffle_idx]
            shuffled_i = i_arr[shuffle_idx]

            # Mini-batch SGD iteration
            for start_b in range(0, n_interactions, self.batch_size):
                end_b = min(start_b + self.batch_size, n_interactions)
                batch_u = shuffled_u[start_b:end_b]
                batch_i = shuffled_i[start_b:end_b]
                b_size = len(batch_u)

                # Vectorized negative sampling with fast rejection
                batch_j = rng.integers(0, num_songs, size=b_size, dtype=np.int32)
                for idx in range(b_size):
                    u_val = batch_u[idx]
                    pos_set = user_positives[u_val]
                    while batch_j[idx] in pos_set:
                        batch_j[idx] = rng.integers(0, num_songs)

                # Compute predictions: x_ui and x_uj
                u_vecs = self.user_factors[batch_u]  # (B, k)
                i_vecs = self.item_factors[batch_i]  # (B, k)
                j_vecs = self.item_factors[batch_j]  # (B, k)

                x_ui = np.sum(u_vecs * i_vecs, axis=1)  # (B,)
                x_uj = np.sum(u_vecs * j_vecs, axis=1)  # (B,)
                x_uij = x_ui - x_uj                     # (B,)

                # BPR loss: -ln(sigmoid(x_uij))
                # Deriv w.r.t x_uij: sigma(-x_uij)
                loss = -np.log(sigmoid(x_uij) + 1e-10)
                epoch_loss += float(np.sum(loss))

                weight = sigmoid(-x_uij)[:, np.newaxis]  # (B, 1)

                # Gradients
                grad_u = weight * (i_vecs - j_vecs) - self.regularization * u_vecs
                grad_i = weight * u_vecs - self.regularization * i_vecs
                grad_j = -weight * u_vecs - self.regularization * j_vecs

                # In-place parameter updates with np.add.at for collision safety
                np.add.at(self.user_factors, batch_u, self.learning_rate * grad_u)
                np.add.at(self.item_factors, batch_i, self.learning_rate * grad_i)
                np.add.at(self.item_factors, batch_j, self.learning_rate * grad_j)

            avg_loss = epoch_loss / n_interactions
            self.training_losses.append(avg_loss)
            epoch_time = time.time() - epoch_start

            if epoch % 5 == 0 or epoch == 1 or epoch == self.n_epochs:
                logger.info(f"BPR-MF Epoch {epoch:02d}/{self.n_epochs} - Loss: {avg_loss:.4f} ({epoch_time:.2f}s)")

        elapsed = time.time() - start_time
        self.is_fitted = True
        logger.info(f"Fitted {self.name} successfully in {elapsed:.2f}s. Final loss: {self.training_losses[-1]:.4f}")
        return self

    def score(self, user_id: str, song_ids: Sequence[str]) -> np.ndarray:
        """Compute inner product scores between user factor and candidate item factors."""
        if not self.is_fitted or self.user_factors is None or self.item_factors is None:
            raise RuntimeError(f"Model '{self.name}' must be fitted before calling score().")

        if user_id not in self.user_to_idx:
            return np.zeros(len(song_ids), dtype=np.float32)

        u_idx = self.user_to_idx[user_id]
        u_factor = self.user_factors[u_idx]

        scores = np.zeros(len(song_ids), dtype=np.float32)
        for idx, song_id in enumerate(song_ids):
            if song_id in self.song_to_idx:
                s_idx = self.song_to_idx[song_id]
                scores[idx] = float(np.dot(u_factor, self.item_factors[s_idx]))

        return scores
