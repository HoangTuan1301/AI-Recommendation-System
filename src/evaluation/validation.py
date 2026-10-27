"""Validation splitting harness for hyperparameter selection in Phase 4B.

This module creates a strictly disjoint, reproducible Train-Inner and Validation
partition from `train.csv` (MovieLens 100K 80k train set).

Guarantees:
1. Seed 42 for 100% deterministic reproducibility.
2. Stratified per-user split (80% train-inner, 20% validation).
3. Anchor mechanism: all movies present in train.csv have at least one interaction
   in train_inner.csv to avoid unhandled cold items during validation ranking.
4. Complete isolation: test.csv is never accessed or touched.
5. Disjointness: (user_id, movie_id) in val.csv never appears in train_inner.csv.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Any, Dict, Tuple
import json
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RANDOM_STATE: int = 42
DEFAULT_VAL_RATIO: float = 0.2


def split_train_val_per_user(
    train_df: pd.DataFrame,
    val_ratio: float = DEFAULT_VAL_RATIO,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Perform a stratified per-user split on the training set to create validation data.

    Args:
        train_df: Training DataFrame containing ['user_id', 'movie_id', 'rating', 'timestamp'].
        val_ratio: Fraction of interactions per user allocated to validation (default 0.20).
        random_state: Fixed random seed (default 42).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
            (train_inner_df, val_df, metadata_dict)
    """
    logger.info("Splitting train.csv into Train-Inner and Validation (val_ratio=%.2f, seed=%d)...", val_ratio, random_state)
    
    # Deterministic shuffle
    shuffled = train_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # Item anchoring: reserve first occurrence of each movie for train_inner
    first_seen_movie_indices = set()
    seen_movies = set()
    for idx, row in shuffled.iterrows():
        mid = row["movie_id"]
        if mid not in seen_movies:
            seen_movies.add(mid)
            first_seen_movie_indices.add(idx)

    val_indices = []
    train_inner_indices = []

    for _, group in shuffled.groupby("user_id", sort=False):
        n_total = len(group)
        n_val = int(round(val_ratio * n_total))

        # Eligible candidates for validation are those not anchored
        group_indices = group.index.tolist()
        val_eligible = [idx for idx in group_indices if idx not in first_seen_movie_indices]

        # Allocate exactly n_val (or maximum eligible)
        selected_val = val_eligible[:n_val]
        selected_val_set = set(selected_val)

        for idx in group_indices:
            if idx in selected_val_set:
                val_indices.append(idx)
            else:
                train_inner_indices.append(idx)

    train_inner_df = shuffled.loc[train_inner_indices].sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    val_df = shuffled.loc[val_indices].sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # Verification checks
    train_pairs = set(zip(train_inner_df["user_id"], train_inner_df["movie_id"]))
    val_pairs = set(zip(val_df["user_id"], val_df["movie_id"]))
    overlap = train_pairs.intersection(val_pairs)
    assert len(overlap) == 0, f"Critical Leakage: {len(overlap)} interactions overlap between Train-Inner and Val!"

    metadata = {
        "train_csv_total_rows": len(train_df),
        "train_inner_rows": len(train_inner_df),
        "val_rows": len(val_df),
        "train_inner_ratio": len(train_inner_df) / len(train_df),
        "val_ratio": len(val_df) / len(train_df),
        "unique_users_train_inner": int(train_inner_df["user_id"].nunique()),
        "unique_users_val": int(val_df["user_id"].nunique()),
        "unique_movies_train_inner": int(train_inner_df["movie_id"].nunique()),
        "unique_movies_val": int(val_df["movie_id"].nunique()),
        "random_state": random_state,
        "is_disjoint": True,
    }

    logger.info(
        "Validation split complete: Train-Inner=%d (%.2f%%), Val=%d (%.2f%%)",
        metadata["train_inner_rows"],
        metadata["train_inner_ratio"] * 100,
        metadata["val_rows"],
        metadata["val_ratio"] * 100,
    )
    return train_inner_df, val_df, metadata


def get_or_create_validation_split() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Load existing Train-Inner / Val split or generate and save if absent.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]: (train_inner_df, val_df, metadata)
    """
    train_inner_path = DATA_PROCESSED_DIR / "train_inner.csv"
    val_path = DATA_PROCESSED_DIR / "val.csv"
    metadata_path = DATA_PROCESSED_DIR / "split_inner_metadata.json"

    if train_inner_path.exists() and val_path.exists() and metadata_path.exists():
        logger.info("Loading existing validation split from %s", DATA_PROCESSED_DIR)
        train_inner_df = pd.read_csv(train_inner_path)
        val_df = pd.read_csv(val_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return train_inner_df, val_df, metadata

    train_df = pd.read_csv(DATA_PROCESSED_DIR / "train.csv")
    train_inner_df, val_df, metadata = split_train_val_per_user(train_df)

    train_inner_df.to_csv(train_inner_path, index=False)
    val_df.to_csv(val_path, index=False)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved validation artifacts to %s", DATA_PROCESSED_DIR)
    return train_inner_df, val_df, metadata


if __name__ == "__main__":
    get_or_create_validation_split()
