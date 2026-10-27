"""Content-Based Filtering Recommender Baseline.

This model builds vector representations of items from metadata (genres and titles)
using TF-IDF feature extraction, and constructs personalized user preference profiles
from historical ratings in the Training set.

Key design principles:
1. Item Representations:
   - Movie metadata (19 genres + movie title tokens) is vectorized using TF-IDF.
   - Item vectors are L2-normalized, allowing cosine similarity to be computed via dot product.
2. User Profile Construction (Strictly from Train):
   - For user u with training ratings r_{ui}, rating-centered weights are computed:
         w_{ui} = r_{ui} - r_u_mean
   - The user profile is computed via vectorized sparse-dense matrix multiplication:
         P = W * V
     where W is the sparse centered rating matrix (num_users x num_movies),
     and V is the item feature matrix (num_movies x num_features).
   - User profiles are L2-normalized: p_u = p_u / ||p_u||_2.
3. Ranking:
   - Score for candidate item i: score(u, i) = dot(p_u, v_i).
   - Candidates are ranked by score descending, movie_id ascending.
4. Strict Isolation:
   - Only Train interactions are used to learn user profiles. Test data is never observed.
"""

from pathlib import Path
import sys

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Dict, List, Optional, Sequence, Union
import logging
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.models.base_recommender import BaseRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ContentBasedRecommender(BaseRecommender):
    """Vectorized Content-Based Recommender utilizing TF-IDF and Cosine Similarity."""

    def __init__(
        self,
        movies_df: pd.DataFrame,
        name: str = "ContentBasedRecommender",
        min_df: int = 1,
    ) -> None:
        """Initialize ContentBasedRecommender.

        Args:
            movies_df: DataFrame containing movie metadata ['movie_id', 'title', 'genres_str'].
            name: Model identifier.
            min_df: Minimum term frequency for TF-IDF vectorizer.
        """
        super().__init__(name=name)
        self.movies_df = movies_df.copy()
        self.min_df = min_df

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.item_features: Optional[np.ndarray] = None  # shape (max_movie_id + 1, num_features)
        self.movie_id_to_idx: Dict[int, int] = {}
        self.user_profiles: Optional[np.ndarray] = None  # shape (max_user_id + 1, num_features)
        self.global_fallback_profile: Optional[np.ndarray] = None
        self.num_features: int = 0
        self.max_movie_id: int = 0
        self.max_user_id: int = 0

        self._build_item_representations()

    def _build_item_representations(self) -> None:
        """Construct TF-IDF feature matrix from static public movie metadata."""
        logger.info("Extracting TF-IDF metadata representations for %d movies...", len(self.movies_df))

        def format_content(row: pd.Series) -> str:
            genres_expanded = " ".join(row["genres_str"].split("|"))
            # Repeat genres 3 times to provide balanced domain weight against title tokens
            title = row["title"]
            return f"{genres_expanded} {genres_expanded} {genres_expanded} {title}"

        corpus = self.movies_df.apply(format_content, axis=1)

        self.vectorizer = TfidfVectorizer(
            token_pattern=r"(?u)\b[\w\'-]+\b",
            min_df=self.min_df,
            stop_words="english",
        )
        raw_tfidf = self.vectorizer.fit_transform(corpus).toarray().astype(np.float32)
        norm_tfidf = normalize(raw_tfidf, norm="l2", axis=1)

        self.num_features = norm_tfidf.shape[1]
        self.max_movie_id = int(self.movies_df["movie_id"].max())

        # Direct-indexable item features table of shape (max_movie_id + 1, num_features)
        self.item_features = np.zeros((self.max_movie_id + 1, self.num_features), dtype=np.float32)
        for idx, mid in enumerate(self.movies_df["movie_id"]):
            self.item_features[int(mid)] = norm_tfidf[idx]
            self.movie_id_to_idx[int(mid)] = idx

        logger.info(
            "Item representations built: item_features shape %s across %d vocabulary terms.",
            self.item_features.shape,
            len(self.vectorizer.vocabulary_),
        )

    def fit(self, train_df: pd.DataFrame) -> "ContentBasedRecommender":
        """Build user preference profiles strictly from training interactions.

        Args:
            train_df: Training DataFrame containing ['user_id', 'movie_id', 'rating'].

        Returns:
            self: The fitted recommender.
        """
        logger.info("Fitting %s on %d training interactions...", self.name, len(train_df))
        self._store_train_metadata(train_df)

        df = train_df.copy()
        user_means = df.groupby("user_id")["rating"].transform("mean")
        df["centered_weight"] = (df["rating"] - user_means).astype(np.float32)

        self.max_user_id = int(df["user_id"].max())

        # Construct sparse interaction weight matrix W of shape (max_user_id + 1, max_movie_id + 1)
        w_sparse = csr_matrix(
            (df["centered_weight"], (df["user_id"], df["movie_id"])),
            shape=(self.max_user_id + 1, self.max_movie_id + 1),
            dtype=np.float32,
        )

        # High-performance sparse-dense dot product: (U, M) @ (M, D) -> (U, D)
        unnorm_profiles = w_sparse.dot(self.item_features)

        # L2 normalize user profiles
        self.user_profiles = normalize(unnorm_profiles, norm="l2", axis=1)

        # Global fallback profile for cold users (mean across trained users)
        trained_mask = np.isin(np.arange(self.max_user_id + 1), list(self.all_train_users))
        if trained_mask.any():
            avg_profile = np.mean(self.user_profiles[trained_mask], axis=0, keepdims=True)
            self.global_fallback_profile = normalize(avg_profile, norm="l2", axis=1).squeeze()
        else:
            self.global_fallback_profile = np.zeros(self.num_features, dtype=np.float32)

        logger.info("Fitted %d user preference profiles.", len(self.all_train_users))
        return self

    def recommend(
        self,
        user_id: int,
        top_n: int = 10,
        candidate_items: Optional[Sequence[int]] = None,
    ) -> List[int]:
        """Generate top-N content recommendations for user_id.

        Args:
            user_id: User identifier.
            top_n: Number of recommendations to generate.
            candidate_items: Candidate movie IDs to score. If None, uses all unseen catalog items.

        Returns:
            List[int]: Top-N recommended movie IDs.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit(train_df) first.")

        user_seen = self.user_train_history.get(user_id, set())

        # Retrieve user profile vector
        if 0 <= user_id <= self.max_user_id and user_id in self.all_train_users:
            profile = self.user_profiles[user_id]
        else:
            profile = self.global_fallback_profile

        # Candidate item list
        if candidate_items is not None:
            eval_candidates = [
                mid for mid in candidate_items
                if mid not in user_seen and 0 <= mid <= self.max_movie_id
            ]
        else:
            eval_candidates = [
                mid for mid in self.all_train_items
                if mid not in user_seen and 0 <= mid <= self.max_movie_id
            ]

        if not eval_candidates:
            return []

        # Vectorized dot product for candidate items
        cand_indices = np.array(eval_candidates, dtype=int)
        cand_vectors = self.item_features[cand_indices]  # (K, D)
        scores = np.dot(cand_vectors, profile)           # (K,)

        # Sort by score desc, then movie_id asc
        scored_pairs = list(zip(eval_candidates, scores))
        scored_pairs.sort(key=lambda x: (-x[1], x[0]))

        return [mid for mid, _ in scored_pairs[:top_n]]

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict cosine similarity preference score between user profile and movie.

        Args:
            user_id: User identifier.
            movie_id: Movie identifier.

        Returns:
            float: Cosine similarity score in [-1.0, 1.0].
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit(train_df) first.")
        if movie_id < 0 or movie_id > self.max_movie_id:
            return 0.0

        if 0 <= user_id <= self.max_user_id and user_id in self.all_train_users:
            profile = self.user_profiles[user_id]
        else:
            profile = self.global_fallback_profile

        item_vec = self.item_features[movie_id]
        return float(np.dot(profile, item_vec))
