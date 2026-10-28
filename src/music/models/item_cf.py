"""Item-Based Collaborative Filtering Recommender for Implicit Music Data.

Computes Cosine Similarity on sparse song-user interaction profiles,
maintaining Top-K nearest neighbors per item for real-time scoring.
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


class ItemCFMusicRecommender(BaseMusicRecommender):
    """Item-Based Collaborative Filtering using Cosine Similarity on implicit listening vectors."""

    def __init__(
        self,
        name: str = "Item-Based CF",
        n_neighbors: int = 50,
        alpha: float = 40.0,
        min_similarity: float = 0.01,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, n_neighbors=n_neighbors, alpha=alpha, min_similarity=min_similarity, **kwargs)
        self.n_neighbors: int = n_neighbors
        self.alpha: float = alpha
        self.min_similarity: float = min_similarity

        self.song_to_idx: Dict[str, int] = {}
        self.idx_to_song: Dict[int, str] = {}
        self.user_to_idx: Dict[str, int] = {}
        self.idx_to_user: Dict[int, str] = {}

        # item_similarities[i] = {j: sim_val} for top-k neighbors
        self.item_similarities: Dict[int, Dict[int, float]] = {}
        # user_interactions[u_idx] = {song_idx: confidence_weight}
        self.user_confidences: Dict[int, Dict[int, float]] = {}
        self.global_popularity: Dict[str, float] = {}

    def fit(self, train_df: pd.DataFrame) -> "ItemCFMusicRecommender":
        """Compute item-item similarity graph over training interactions."""
        self._init_train_state(train_df)

        unique_users = sorted(train_df["user_id"].unique())
        unique_songs = self.all_train_items

        self.user_to_idx = {u: i for i, u in enumerate(unique_users)}
        self.idx_to_user = {i: u for i, u in enumerate(unique_users)}
        self.song_to_idx = {s: i for i, s in enumerate(unique_songs)}
        self.idx_to_song = {i: s for i, s in enumerate(unique_songs)}

        num_users = len(unique_users)
        num_songs = len(unique_songs)

        # Build user-song confidence map and sparse item-user matrix
        u_indices = np.array([self.user_to_idx[u] for u in train_df["user_id"]], dtype=np.int32)
        s_indices = np.array([self.song_to_idx[s] for s in train_df["song_id"]], dtype=np.int32)
        play_counts = train_df["play_count"].to_numpy(dtype=np.float32)

        confidences = 1.0 + self.alpha * np.log1p(play_counts)

        # Build user_confidences lookup
        self.user_confidences = {u_idx: {} for u_idx in range(num_users)}
        for u_idx, s_idx, c_val in zip(u_indices, s_indices, confidences):
            self.user_confidences[u_idx][s_idx] = float(c_val)

        # Sparse Item-User Matrix of shape (num_songs, num_users)
        item_user_matrix = csr_matrix(
            (confidences, (s_indices, u_indices)),
            shape=(num_songs, num_users),
            dtype=np.float32,
        )

        logger.info(f"Computing pairwise Cosine similarity across {num_songs:,} items...")
        sim_matrix = cosine_similarity(item_user_matrix, dense_output=True)

        # Filter top-K neighbors per item
        self.item_similarities = {}
        for i in range(num_songs):
            row = sim_matrix[i]
            row[i] = 0.0  # Zero self-similarity
            # Select top-N neighbor indices
            if np.any(row > self.min_similarity):
                top_neighbors = np.argsort(-row)[: self.n_neighbors]
                self.item_similarities[i] = {
                    int(j): float(row[j])
                    for j in top_neighbors
                    if row[j] >= self.min_similarity
                }
            else:
                self.item_similarities[i] = {}

        # Compute popularity fallback
        song_counts = train_df.groupby("song_id")["user_id"].count()
        self.global_popularity = dict(song_counts / song_counts.max())

        self.is_fitted = True
        logger.info(f"Fitted {self.name}: Top-{self.n_neighbors} neighbor graph constructed.")
        return self

    def score(self, user_id: str, song_ids: Sequence[str]) -> np.ndarray:
        """Compute predicted preference score for candidate songs.

        score(u, i) = sum_{j in H_u cap N(i)} sim(i, j) * c_uj
        """
        if not self.is_fitted:
            raise RuntimeError(f"Model '{self.name}' must be fitted before calling score().")

        if user_id not in self.user_to_idx:
            # Fallback for cold users
            return np.array([self.global_popularity.get(s, 0.0) * 0.01 for s in song_ids], dtype=np.float32)

        u_idx = self.user_to_idx[user_id]
        user_history = self.user_confidences.get(u_idx, {})
        if not user_history:
            return np.array([self.global_popularity.get(s, 0.0) * 0.01 for s in song_ids], dtype=np.float32)

        scores = np.zeros(len(song_ids), dtype=np.float32)

        for idx, song_id in enumerate(song_ids):
            if song_id not in self.song_to_idx:
                continue
            i_idx = self.song_to_idx[song_id]
            neighbors = self.item_similarities.get(i_idx, {})
            if not neighbors:
                continue

            # Accumulate similarity * confidence
            score_acc = 0.0
            for j_idx, sim_val in neighbors.items():
                if j_idx in user_history:
                    score_acc += sim_val * user_history[j_idx]

            scores[idx] = score_acc

        return scores
