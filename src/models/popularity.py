"""Popularity-based Recommender Baseline.

This model ranks movies strictly based on aggregate statistics derived from the
Training set. It recommends the highest-scoring unseen items to each user.

Supported scoring methods:
1. 'weighted_rating' (IMDb Bayesian weighted rating, default):
       WR = (v / (v + m)) * R + (m / (v + m)) * C
   where:
       v: number of ratings for the movie in Train
       m: minimum threshold (default: 25th percentile of rating counts)
       R: mean rating of the movie in Train
       C: global mean rating across the entire training set
2. 'rating_count' (Pure popularity based on frequency of engagement):
       Score = v

Tie-breaking: Deterministic ordering by score descending, then movie_id ascending.
"""

from pathlib import Path
import sys

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Dict, List, Optional, Sequence, Tuple
import logging
import pandas as pd

from src.models.base_recommender import BaseRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PopularityRecommender(BaseRecommender):
    """Popularity-based recommendation model."""

    def __init__(
        self,
        name: str = "PopularityRecommender",
        method: str = "weighted_rating",
        min_ratings_threshold: Optional[float] = None,
    ) -> None:
        """Initialize the PopularityRecommender.

        Args:
            name: Human-readable model identifier.
            method: 'weighted_rating' (IMDb Bayesian average) or 'rating_count'.
            min_ratings_threshold: Damping parameter m for Bayesian average.
                If None, defaults to the 25th percentile of item rating counts.
        """
        super().__init__(name=name)
        if method not in ("weighted_rating", "rating_count"):
            raise ValueError(f"Unknown method '{method}'. Choose 'weighted_rating' or 'rating_count'.")

        self.method: str = method
        self.min_ratings_threshold: Optional[float] = min_ratings_threshold
        self.item_scores: Dict[int, float] = {}
        self.global_mean: float = 0.0
        self.ranked_catalog: List[Tuple[int, float]] = []

    def fit(self, train_df: pd.DataFrame) -> "PopularityRecommender":
        """Compute popularity scores strictly from the training dataset.

        Args:
            train_df: Training DataFrame containing ['user_id', 'movie_id', 'rating'].

        Returns:
            self: The fitted recommender.
        """
        logger.info("Fitting %s using method='%s' on %d training rows...", self.name, self.method, len(train_df))
        self._store_train_metadata(train_df)

        self.global_mean = float(train_df["rating"].mean())

        # Aggregate item counts and averages strictly from Train
        item_stats = (
            train_df.groupby("movie_id")
            .agg(v=("rating", "count"), R=("rating", "mean"))
            .reset_index()
        )

        if self.method == "rating_count":
            item_stats["score"] = item_stats["v"].astype(float)
        else:  # weighted_rating
            m = (
                self.min_ratings_threshold
                if self.min_ratings_threshold is not None
                else float(item_stats["v"].quantile(0.25))
            )
            c = self.global_mean
            item_stats["score"] = (item_stats["v"] / (item_stats["v"] + m)) * item_stats["R"] + (
                m / (item_stats["v"] + m)
            ) * c

        # Store dictionary of scores
        self.item_scores = dict(zip(item_stats["movie_id"].astype(int), item_stats["score"].astype(float)))

        # Build globally ranked list: sorted by score descending, then movie_id ascending for deterministic tie-break
        sorted_stats = item_stats.sort_values(by=["score", "movie_id"], ascending=[False, True])
        self.ranked_catalog = list(
            zip(sorted_stats["movie_id"].astype(int), sorted_stats["score"].astype(float))
        )

        logger.info(
            "Fitted %d movies. Top 3 popular movies: %s",
            len(self.item_scores),
            self.ranked_catalog[:3],
        )
        return self

    def recommend(
        self,
        user_id: int,
        top_n: int = 10,
        candidate_items: Optional[Sequence[int]] = None,
    ) -> List[int]:
        """Recommend the most popular unseen items for user_id.

        Args:
            user_id: User identifier.
            top_n: Number of recommendations to produce.
            candidate_items: Optional candidate item IDs. If None, uses all unseen items.

        Returns:
            List[int]: Top-N recommended movie IDs.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit(train_df) first.")

        user_seen = self.user_train_history.get(user_id, set())

        if candidate_items is not None:
            candidate_set = set(candidate_items)
            # Filter and rank candidates according to precomputed score
            scored_candidates = [
                (mid, self.item_scores.get(mid, -1.0))
                for mid in candidate_set
                if mid not in user_seen
            ]
            # Sort by score desc, then movie_id asc
            scored_candidates.sort(key=lambda x: (-x[1], x[0]))
            return [mid for mid, _ in scored_candidates[:top_n]]

        # Default: pick from pre-sorted ranked catalog excluding seen items
        recommendations: List[int] = []
        for mid, _ in self.ranked_catalog:
            if mid not in user_seen:
                recommendations.append(mid)
                if len(recommendations) == top_n:
                    break
        return recommendations

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict preference score (movie's aggregate score or global mean if unseen).

        Args:
            user_id: User identifier.
            movie_id: Movie identifier.

        Returns:
            float: Popularity score.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit(train_df) first.")
        return float(self.item_scores.get(movie_id, self.global_mean))
