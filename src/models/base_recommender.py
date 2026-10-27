"""Abstract Base Class for Recommender Systems in AI-Recommendation-System.

All recommendation models inherit from BaseRecommender to ensure strict adherence
to the unified interface required for benchmark evaluation:
- fit(train_data)
- recommend(user_id, top_n=10, candidate_items=None)
- predict(user_id, movie_id) [optional, where supported]
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Set, Union
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BaseRecommender(ABC):
    """Abstract Base Class defining the contract for all 10 recommendation models."""

    def __init__(self, name: Optional[str] = None) -> None:
        """Initialize the base recommender.

        Args:
            name: Human-readable name of the model.
        """
        self.name: str = name or self.__class__.__name__
        self.is_fitted: bool = False
        self.user_train_history: Dict[int, Set[int]] = {}
        self.all_train_items: Set[int] = set()
        self.all_train_users: Set[int] = set()

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> "BaseRecommender":
        """Fit the recommendation model strictly on training data.

        Args:
            train_df: DataFrame containing at least ['user_id', 'movie_id', 'rating'].

        Returns:
            self: The fitted instance.
        """
        pass

    @abstractmethod
    def recommend(
        self,
        user_id: int,
        top_n: int = 10,
        candidate_items: Optional[Sequence[int]] = None,
    ) -> List[int]:
        """Generate top_n recommendations for a given user.

        Args:
            user_id: Target user ID.
            top_n: Number of items to recommend (default 10).
            candidate_items: Optional list of candidate item IDs to rank.
                If None, candidates default to all catalog items minus user's train history.

        Returns:
            List[int]: Ranked list of unique recommended movie_id values (length <= top_n).
        """
        pass

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict the rating or preference score for a user-item pair.

        Args:
            user_id: User ID.
            movie_id: Movie ID.

        Returns:
            float: Predicted score or rating.

        Raises:
            NotImplementedError: If the model does not support individual rating prediction.
        """
        raise NotImplementedError(f"Model {self.name} does not support point-wise predict().")

    def _store_train_metadata(self, train_df: pd.DataFrame) -> None:
        """Utility to capture user train history and catalog IDs from training data.

        Args:
            train_df: DataFrame containing ['user_id', 'movie_id', 'rating'].
        """
        self.user_train_history = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()
        self.all_train_items = set(train_df["movie_id"].unique())
        self.all_train_users = set(train_df["user_id"].unique())
        self.is_fitted = True

    def _get_default_candidates(self, user_id: int) -> List[int]:
        """Get candidate items for user by excluding seen items in training set.

        Args:
            user_id: User ID.

        Returns:
            List[int]: Unseen candidate item IDs.
        """
        seen = self.user_train_history.get(user_id, set())
        return [item for item in self.all_train_items if item not in seen]
