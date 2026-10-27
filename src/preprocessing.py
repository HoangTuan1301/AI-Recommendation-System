"""Data preprocessing, cleaning, train/test splitting, and sparse matrix generation.

This module prepares clean datasets for the 10 recommendation models:
- Stratified per-user 80/20 train/test splitting with fixed random seed 42.
- Positive interaction tagging (rating >= 4) for Top-N ranking metrics.
- User and item matrix index mappings computed from training interactions.
- Compressed Sparse Row (CSR) User-Item interaction matrix construction.
"""

from pathlib import Path
import sys

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Any, Dict, Optional, Tuple, Union
import json
import logging
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.data_loader import GENRE_NAMES, load_movielens_100k, validate_movielens_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RELEVANT_RATING_THRESHOLD: float = 4.0
RANDOM_STATE: int = 42
DEFAULT_TEST_RATIO: float = 0.2


def clean_ratings(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean ratings DataFrame.

    Ensures proper dtypes, range [1, 5], removes duplicates,
    and adds binary relevance indicator (rating >= 4).

    Args:
        ratings_df: Raw ratings DataFrame.

    Returns:
        pd.DataFrame: Cleaned ratings DataFrame.
    """
    df = ratings_df.copy()
    df = df.dropna(subset=["user_id", "movie_id", "rating"])
    df["user_id"] = df["user_id"].astype(int)
    df["movie_id"] = df["movie_id"].astype(int)
    df["rating"] = df["rating"].astype(float)
    df["timestamp"] = df["timestamp"].astype(int)

    # Filter out bounds if any (MovieLens 100K is 1-5)
    df = df[(df["rating"] >= 1.0) & (df["rating"] <= 5.0)]

    # Deduplicate keeping last timestamp if any duplicates exist
    df = df.sort_values("timestamp").drop_duplicates(subset=["user_id", "movie_id"], keep="last")

    # Add relevance flag for Top-N ranking without dropping non-relevant items
    df["is_relevant"] = (df["rating"] >= RELEVANT_RATING_THRESHOLD).astype(int)
    return df.reset_index(drop=True)


def clean_movies(movies_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean movies metadata.

    Args:
        movies_df: Raw movies DataFrame.

    Returns:
        pd.DataFrame: Cleaned movies metadata DataFrame.
    """
    df = movies_df.copy()
    df["movie_id"] = df["movie_id"].astype(int)
    df["title"] = df["title"].astype(str).str.strip()

    # Extract release year from title if present, e.g. "Toy Story (1995)"
    df["year"] = df["title"].str.extract(r"\((\d{4})\)").astype(float)

    # Clean release_date
    df["release_date"] = pd.to_datetime(df["release_date"], format="%d-%b-%Y", errors="coerce")

    # Fill genre columns
    for g in GENRE_NAMES:
        if g in df.columns:
            df[g] = df[g].fillna(0).astype(int)

    # Generate string representation of genres for easy reference/filtering
    def extract_genre_str(row: pd.Series) -> str:
        active = [g for g in GENRE_NAMES if g != "unknown" and row[g] == 1]
        return "|".join(active) if active else "unknown"

    df["genres_str"] = df.apply(extract_genre_str, axis=1)
    return df


def split_train_test_per_user(
    ratings_df: pd.DataFrame,
    test_ratio: float = DEFAULT_TEST_RATIO,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Perform a stratified per-user 80/20 train/test split.

    Guarantees:
    1. Deterministic and fully reproducible via fixed seed.
    2. Every user in Test has at least 80% of their rating history in Train.
    3. Disjoint partition: no interaction (user_id, movie_id) appears in both sets.
    4. Test movies have representation in Train to prevent unhandled cold-start.
    5. Preserves all rating levels (1 to 5).

    Args:
        ratings_df: Cleaned ratings DataFrame.
        test_ratio: Target fraction of test interactions per user (default 0.2).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]: (train_df, test_df, metadata_dict)
    """
    logger.info("Starting per-user train/test split (ratio=%.2f, seed=%d)...", test_ratio, random_state)

    # Deterministic global shuffle
    shuffled = ratings_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # To guarantee every movie appearing in test has at least 1 rating in train:
    # We reserve the first occurrence of each movie in the shuffled table for Train.
    first_seen_movie_indices = set()
    seen_movies = set()
    for idx, row in shuffled.iterrows():
        mid = row["movie_id"]
        if mid not in seen_movies:
            seen_movies.add(mid)
            first_seen_movie_indices.add(idx)

    test_indices = []
    train_indices = []

    for _, group in shuffled.groupby("user_id", sort=False):
        n_total = len(group)
        n_test = int(round(test_ratio * n_total))

        # Test candidates cannot include the reserved train anchors
        candidates = [idx for idx in group.index if idx not in first_seen_movie_indices]

        u_test = candidates[:n_test]
        u_train = [idx for idx in group.index if idx not in u_test]

        test_indices.extend(u_test)
        train_indices.extend(u_train)

    train_df = shuffled.loc[train_indices].copy().sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    test_df = shuffled.loc[test_indices].copy().sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # Integrity verification
    assert len(set(train_indices).intersection(set(test_indices))) == 0, "Train and Test sets are not disjoint!"
    assert set(test_df["user_id"]).issubset(set(train_df["user_id"])), "Not all test users are in train!"
    assert set(test_df["movie_id"]).issubset(set(train_df["movie_id"])), "Cold-start items detected in test!"

    metadata = {
        "random_state": random_state,
        "target_test_ratio": test_ratio,
        "total_ratings": len(ratings_df),
        "train_ratings": len(train_df),
        "test_ratings": len(test_df),
        "train_percentage": round(len(train_df) / len(ratings_df) * 100, 2),
        "test_percentage": round(len(test_df) / len(ratings_df) * 100, 2),
        "num_users_train": int(train_df["user_id"].nunique()),
        "num_users_test": int(test_df["user_id"].nunique()),
        "num_movies_train": int(train_df["movie_id"].nunique()),
        "num_movies_test": int(test_df["movie_id"].nunique()),
        "relevant_threshold": RELEVANT_RATING_THRESHOLD,
        "train_relevant_ratings": int(train_df["is_relevant"].sum()),
        "test_relevant_ratings": int(test_df["is_relevant"].sum()),
        "cold_start_policy": "All test items have prior training history; baseline/content fallback for unobserved items.",
    }

    logger.info("Split completed: %d Train (%.2f%%) / %d Test (%.2f%%).",
                len(train_df), metadata["train_percentage"], len(test_df), metadata["test_percentage"])
    return train_df, test_df, metadata


def build_user_item_mappings(
    train_df: pd.DataFrame,
    all_movies_df: Optional[pd.DataFrame] = None,
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int], Dict[int, int]]:
    """Build contiguous 0-indexed matrix mappings for users and movies from Train.

    Args:
        train_df: Training ratings DataFrame.
        all_movies_df: Optional full movie catalog to include full movie set.

    Returns:
        Tuple[Dict[int, int], Dict[int, int], Dict[int, int], Dict[int, int]]:
            (user_to_idx, idx_to_user, movie_to_idx, idx_to_movie)
    """
    unique_users = sorted(train_df["user_id"].unique())
    user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
    idx_to_user = {i: uid for uid, i in user_to_idx.items()}

    if all_movies_df is not None:
        unique_movies = sorted(all_movies_df["movie_id"].unique())
    else:
        unique_movies = sorted(train_df["movie_id"].unique())

    movie_to_idx = {mid: j for j, mid in enumerate(unique_movies)}
    idx_to_movie = {j: mid for mid, j in movie_to_idx.items()}

    return user_to_idx, idx_to_user, movie_to_idx, idx_to_movie


def create_sparse_user_item_matrix(
    ratings_df: pd.DataFrame,
    user_to_idx: Dict[int, int],
    movie_to_idx: Dict[int, int],
) -> csr_matrix:
    """Construct a Compressed Sparse Row (CSR) matrix of user-item ratings.

    Note:
        Unobserved ratings are NOT represented as 0; they are empty sparse entries.
        Only explicitly observed ratings (1 to 5) occupy matrix entries.

    Args:
        ratings_df: Ratings DataFrame containing [user_id, movie_id, rating].
        user_to_idx: Mapping from user_id to row index.
        movie_to_idx: Mapping from movie_id to column index.

    Returns:
        csr_matrix: Sparse matrix of shape (num_users, num_movies).
    """
    valid_ratings = ratings_df[
        ratings_df["user_id"].isin(user_to_idx) & ratings_df["movie_id"].isin(movie_to_idx)
    ]

    row_indices = [user_to_idx[uid] for uid in valid_ratings["user_id"]]
    col_indices = [movie_to_idx[mid] for mid in valid_ratings["movie_id"]]
    values = valid_ratings["rating"].values.astype(np.float32)

    shape = (len(user_to_idx), len(movie_to_idx))
    sparse_matrix = csr_matrix((values, (row_indices, col_indices)), shape=shape, dtype=np.float32)
    return sparse_matrix


def run_full_preprocessing_pipeline(
    raw_data_dir: Union[str, Path] = "data/raw/ml-100k",
    output_dir: Union[str, Path] = "data/processed",
) -> Dict[str, Any]:
    """Execute the full data ingestion, cleaning, splitting, and saving pipeline.

    Args:
        raw_data_dir: Source folder of MovieLens 100K.
        output_dir: Output folder for processed artifacts.

    Returns:
        Dict[str, Any]: Execution metadata summary.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load raw data
    raw_ratings, raw_movies, raw_users = load_movielens_100k(raw_data_dir)
    val_report = validate_movielens_data(raw_ratings, raw_movies, raw_users)

    # 2. Clean data
    clean_r = clean_ratings(raw_ratings)
    clean_m = clean_movies(raw_movies)

    # 3. Train/Test Split
    train_df, test_df, split_meta = split_train_test_per_user(clean_r, test_ratio=0.2, random_state=42)

    # 4. Matrix mappings
    user_to_idx, idx_to_user, movie_to_idx, idx_to_movie = build_user_item_mappings(train_df, clean_m)

    # 5. Build sparse matrix
    train_sparse = create_sparse_user_item_matrix(train_df, user_to_idx, movie_to_idx)

    # 6. Save datasets and metadata
    train_csv = out_path / "train.csv"
    test_csv = out_path / "test.csv"
    movies_clean_csv = out_path / "movies_clean.csv"
    split_meta_json = out_path / "split_metadata.json"
    user_mapping_json = out_path / "user_to_idx.json"
    movie_mapping_json = out_path / "movie_to_idx.json"

    train_df.to_csv(train_csv, index=False, encoding="utf-8")
    test_df.to_csv(test_csv, index=False, encoding="utf-8")
    clean_m.to_csv(movies_clean_csv, index=False, encoding="utf-8")

    split_meta["sparse_matrix_shape"] = list(train_sparse.shape)
    split_meta["sparse_matrix_nnz"] = int(train_sparse.nnz)

    with open(split_meta_json, "w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=4, ensure_ascii=False)

    with open(user_mapping_json, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in user_to_idx.items()}, f, indent=2)

    with open(movie_mapping_json, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in movie_to_idx.items()}, f, indent=2)

    logger.info("Pipeline executed successfully. Processed artifacts saved to: %s", out_path.resolve())
    return split_meta


if __name__ == "__main__":
    meta = run_full_preprocessing_pipeline()
    print("\n--- Preprocessing Pipeline Metadata ---")
    for k, v in meta.items():
        print(f"{k}: {v}")
