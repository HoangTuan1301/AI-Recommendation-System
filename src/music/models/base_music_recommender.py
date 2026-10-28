"""Abstract Base Class for MusicAI Recommendation Models.

All song-level recommendation models inherit from BaseMusicRecommender to enforce
a unified interface:
- fit(train_df)
- recommend(user_id, top_k=10, exclude_seen=True)
- score(user_id, song_ids)
- save(path)
- load(path)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import logging
import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseMusicRecommender(ABC):
    """Abstract Base Class defining the contract for all 6 MusicAI models."""

    def __init__(self, name: Optional[str] = None, **kwargs: Any) -> None:
        self.name: str = name or self.__class__.__name__
        self.is_fitted: bool = False
        self.user_train_history: Dict[str, Set[str]] = {}
        self.all_train_items: List[str] = []
        self.all_train_items_set: Set[str] = set()
        self.all_train_users: Set[str] = set()
        self.hyperparameters: Dict[str, Any] = kwargs
        self.metadata: Dict[str, Any] = {
            "model_name": self.name,
            "version": "1.0.0",
            "framework": "scikit-learn/scipy/pytorch",
        }

    def _init_train_state(self, train_df: pd.DataFrame) -> None:
        """Record user training history and catalog universe from training data."""
        self.user_train_history = (
            train_df.groupby("user_id")["song_id"].apply(set).to_dict()
        )
        self.all_train_items = sorted(train_df["song_id"].unique())
        self.all_train_items_set = set(self.all_train_items)
        self.all_train_users = set(train_df["user_id"].unique())

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> "BaseMusicRecommender":
        """Fit model strictly on training data."""
        pass

    @abstractmethod
    def score(self, user_id: str, song_ids: Sequence[str]) -> np.ndarray:
        """Compute predicted scores for candidate song IDs.

        Args:
            user_id: Raw string user identifier.
            song_ids: Sequence of raw string song identifiers.

        Returns:
            np.ndarray: 1D array of scores corresponding to each song in song_ids.
        """
        pass

    def recommend(
        self,
        user_id: str,
        top_k: int = 10,
        exclude_seen: bool = True,
        candidate_songs: Optional[Sequence[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Generate ranked Top-K song recommendations for a user.

        Args:
            user_id: User identifier.
            top_k: Number of recommendations to return.
            exclude_seen: Whether to filter out songs in user's train history.
            candidate_songs: Optional pool of candidate song IDs. Defaults to all train songs.

        Returns:
            List[Tuple[str, float]]: List of (song_id, score) pairs ranked in descending order.
        """
        if not self.is_fitted:
            raise RuntimeError(f"Model '{self.name}' must be fitted before calling recommend().")

        candidates = list(candidate_songs) if candidate_songs is not None else self.all_train_items

        if exclude_seen and user_id in self.user_train_history:
            seen = self.user_train_history[user_id]
            candidates = [s for s in candidates if s not in seen]

        if not candidates:
            return []

        scores = self.score(user_id, candidates)

        # Handle NaNs or Infs gracefully
        scores = np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)

        # Sort indices in descending order
        top_k = min(top_k, len(candidates))
        best_indices = np.argsort(-scores)[:top_k]

        return [(candidates[idx], float(scores[idx])) for idx in best_indices]

    def save(self, path: Union[str, Path]) -> None:
        """Serialize model instance and metadata to disk."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target_path)
        logger.info(f"Saved model checkpoint to: {target_path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaseMusicRecommender":
        """Load serialized model instance from disk."""
        target_path = Path(path)
        if not target_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {target_path}")
        model = joblib.load(target_path)
        if not isinstance(model, BaseMusicRecommender):
            raise TypeError(f"Loaded object is not a BaseMusicRecommender instance: {type(model)}")
        return model
