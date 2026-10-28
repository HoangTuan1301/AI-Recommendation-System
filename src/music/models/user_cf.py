"""User-Based Collaborative Filtering Recommender for Implicit Music Data.

Identifies Top-K most similar listeners using Cosine Similarity on sparse
interaction profiles and aggregates their listening habits to score unseen songs.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import logging
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from src.music.models.base_music_recommender import BaseMusicRecommender

logger = logging.getLogger(__name__)


class UserCFMusicRecommender(BaseMusicRecommender):
    """User-Based Collaborative Filtering using Cosine Similarity on implicit listening vectors."""

    def __init__(
        self,
        name: str = "User-Based CF",
        n_neighbors: int = 50,
        alpha: float = 40.0,
        min_similarity: float = 0.01,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, n_neighbors=n_neighbors, alpha=alpha, min_similarity=min_similarity, **kwargs)
        self.n_neighbors: int = n_neighbors
        self.alpha: float = alpha
        self.min_similarity: float = min_similarity

        self.user_to_idx: Dict[str, int] = {}
        self.idx_to_user: Dict[int, str] = {}
        self.song_to_idx: Dict[str, int] = {}
        self.idx_to_song: Dict[int, str] = {}

        # user_similarities[u_idx] = {v_idx: sim_val} for top-k neighbors
        self.user_similarities: Dict[int, Dict[int, float]] = {}
        # user_song_matrix: sparse (num_users, num_songs)
        self.user_song_matrix: Optional[csr_matrix] = None
        self.global_popularity: Dict[str, float] = {}

    def fit(self, train_df: pd.DataFrame) -> "UserCFMusicRecommender":
        """Compute user-user similarity graph over training interactions."""
        self._init_train_state(train_df)

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

        confidences = 1.0 + self.alpha * np.log1p(play_counts)

        # Sparse User-Song Matrix of shape (num_users, num_songs)
        self.user_song_matrix = csr_matrix(
            (confidences, (u_indices, s_indices)),
            shape=(num_users, num_songs),
            dtype=np.float32,
        )

        logger.info(f"Computing pairwise Cosine similarity across {num_users:,} users...")
        sim_matrix = cosine_similarity(self.user_song_matrix, dense_output=True)

        self.user_similarities = {}
        for u in range(num_users):
            row = sim_matrix[u]
            row[u] = 0.0  # Zero self-similarity
            if np.any(row > self.min_similarity):
                top_neighbors = np.argsort(-row)[: self.n_neighbors]
                self.user_similarities[u] = {
                    int(v): float(row[v])
                    for v in top_neighbors
                    if row[v] >= self.min_similarity
                }
            else:
                self.user_similarities[u] = {}

        # Compute popularity fallback
        song_counts = train_df.groupby("song_id")["user_id"].count()
        self.global_popularity = dict(song_counts / song_counts.max())

        self.is_fitted = True
        logger.info(f"Fitted {self.name}: Top-{self.n_neighbors} user neighbor graph constructed.")
        return self

    def score(self, user_id: str, song_ids: Sequence[str]) -> np.ndarray:
        """Compute preference scores by aggregating neighbors' listening intensities."""
        if not self.is_fitted or self.user_song_matrix is None:
            raise RuntimeError(f"Model '{self.name}' must be fitted before calling score().")

        if user_id not in self.user_to_idx:
            return np.array([self.global_popularity.get(s, 0.0) * 0.01 for s in song_ids], dtype=np.float32)

        u_idx = self.user_to_idx[user_id]
        neighbors = self.user_similarities.get(u_idx, {})
        if not neighbors:
            return np.array([self.global_popularity.get(s, 0.0) * 0.01 for s in song_ids], dtype=np.float32)

        neighbor_indices = list(neighbors.keys())
        neighbor_weights = np.array([neighbors[v] for v in neighbor_indices], dtype=np.float32)

        # Slice neighbor rows from user_song_matrix
        sub_matrix = self.user_song_matrix[neighbor_indices]  # (n_neighbors, num_songs)
        # Weighted sum: (1, n_neighbors) dot (n_neighbors, num_songs) -> (num_songs,)
        aggregated_scores = sub_matrix.T.dot(neighbor_weights)

        # Map candidate songs to scores
        scores = np.zeros(len(song_ids), dtype=np.float32)
        for idx, song_id in enumerate(song_ids):
            if song_id in self.song_to_idx:
                s_idx = self.song_to_idx[song_id]
                scores[idx] = aggregated_scores[s_idx]

        return scores
