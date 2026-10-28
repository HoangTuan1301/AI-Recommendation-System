"""Popularity-Based Music Recommender Baseline.

Ranks songs based on global listener count and aggregate play volume.
Serves as an essential non-personalized baseline for implicit feedback benchmarking.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import logging
import numpy as np
import pandas as pd

from src.music.models.base_music_recommender import BaseMusicRecommender

logger = logging.getLogger(__name__)


class PopularityMusicRecommender(BaseMusicRecommender):
    """Non-personalized baseline ranking songs by combined listener reach and play volume."""

    def __init__(self, name: str = "Popularity Baseline", **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)
        self.song_scores: Dict[str, float] = {}
        self.ranked_songs: List[str] = []

    def fit(self, train_df: pd.DataFrame) -> "PopularityMusicRecommender":
        """Compute popularity scores across training interactions.

        Formula: score(i) = log(1 + distinct_listeners_i) * log(1 + total_plays_i)
        """
        self._init_train_state(train_df)

        stats = train_df.groupby("song_id").agg(
            distinct_listeners=("user_id", "nunique"),
            total_plays=("play_count", "sum"),
        )

        scores = (
            np.log1p(stats["distinct_listeners"].to_numpy())
            * np.log1p(stats["total_plays"].to_numpy())
        )

        self.song_scores = dict(zip(stats.index, scores))

        # Ensure all catalog items have a score
        for s in self.all_train_items:
            if s not in self.song_scores:
                self.song_scores[s] = 0.0

        # Sort songs by popularity score descending
        self.ranked_songs = sorted(
            self.all_train_items,
            key=lambda s: self.song_scores.get(s, 0.0),
            reverse=True,
        )

        self.is_fitted = True
        logger.info(f"Fitted {self.name}: {len(self.song_scores):,} songs scored.")
        return self

    def score(self, user_id: str, song_ids: Sequence[str]) -> np.ndarray:
        """Return precomputed popularity scores for given songs."""
        if not self.is_fitted:
            raise RuntimeError(f"Model '{self.name}' must be fitted before calling score().")
        return np.array([self.song_scores.get(s, 0.0) for s in song_ids], dtype=np.float32)
