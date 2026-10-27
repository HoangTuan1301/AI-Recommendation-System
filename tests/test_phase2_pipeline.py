"""Comprehensive test suite for Phase 2: Data Ingestion, Preprocessing, and Splitting.

Uses Python standard library `unittest` for zero-dependency execution.

Verifies:
1. Dataset loaded correctly (expected row counts and columns).
2. Ratings are strictly within the valid range [1, 5].
3. Primary keys (user_id, movie_id) have no invalid or missing values.
4. Train and Test interaction sets are strictly disjoint.
5. Every user in Test has sufficient training history.
6. Total interactions conserved (Train + Test == 100,000).
7. Sparse User-Item interaction matrix shape and NNZ match expectations.
8. Processed output files (CSV, JSON) can be read back cleanly.
9. Split is 100% deterministic and reproducible with fixed seed 42.
10. No data leakage from Test into Train (zero overlap, no label contamination).
"""

from pathlib import Path
import json
import sys
import unittest

# Ensure REPO_ROOT is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.data_loader import load_movielens_100k, validate_movielens_data
from src.preprocessing import (
    clean_ratings,
    clean_movies,
    split_train_test_per_user,
    build_user_item_mappings,
    create_sparse_user_item_matrix,
    RELEVANT_RATING_THRESHOLD,
)

PROCESSED_DIR = REPO_ROOT / "data" / "processed"


class TestPhase2Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_ratings, cls.raw_movies, cls.raw_users = load_movielens_100k(
            REPO_ROOT / "data" / "raw" / "ml-100k"
        )
        cls.train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
        cls.test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
        cls.movies_df = pd.read_csv(PROCESSED_DIR / "movies_clean.csv")

    def test_01_dataset_loaded_correctly(self):
        """Criterion 1: Dataset correctly loaded with exact expected dimensions."""
        self.assertEqual(len(self.raw_ratings), 100000, f"Expected 100,000 ratings, got {len(self.raw_ratings)}")
        self.assertEqual(len(self.raw_movies), 1682, f"Expected 1,682 movies, got {len(self.raw_movies)}")
        self.assertEqual(len(self.raw_users), 943, f"Expected 943 users, got {len(self.raw_users)}")
        self.assertEqual(list(self.raw_ratings.columns), ["user_id", "movie_id", "rating", "timestamp"])

    def test_02_ratings_within_valid_domain(self):
        """Criterion 2: Ratings are strictly in range [1, 5]."""
        self.assertGreaterEqual(self.raw_ratings["rating"].min(), 1.0)
        self.assertLessEqual(self.raw_ratings["rating"].max(), 5.0)
        self.assertTrue(self.raw_ratings["rating"].isin([1, 2, 3, 4, 5]).all())

    def test_03_foreign_keys_and_null_checks(self):
        """Criterion 3: No null or orphan user_id / movie_id keys."""
        self.assertEqual(self.raw_ratings["user_id"].isnull().sum(), 0)
        self.assertEqual(self.raw_ratings["movie_id"].isnull().sum(), 0)
        self.assertEqual(self.raw_ratings["rating"].isnull().sum(), 0)

        valid_report = validate_movielens_data(self.raw_ratings, self.raw_movies, self.raw_users)
        self.assertTrue(valid_report["is_valid"])
        self.assertEqual(valid_report["duplicated_interactions"], 0)

    def test_04_train_test_are_strictly_disjoint(self):
        """Criterion 4: No interaction (user_id, movie_id) appears in both Train and Test."""
        train_pairs = set(zip(self.train_df["user_id"], self.train_df["movie_id"]))
        test_pairs = set(zip(self.test_df["user_id"], self.test_df["movie_id"]))

        intersection = train_pairs.intersection(test_pairs)
        self.assertEqual(len(intersection), 0, f"Found {len(intersection)} overlapping interactions!")

    def test_05_every_test_user_has_train_history(self):
        """Criterion 5: Every user in Test has prior history in Train."""
        train_users = set(self.train_df["user_id"])
        test_users = set(self.test_df["user_id"])

        self.assertTrue(test_users.issubset(train_users), "There are users in Test not in Train!")

        user_train_counts = self.train_df.groupby("user_id").size()
        self.assertGreaterEqual(user_train_counts.min(), 16, f"Min train ratings too few: {user_train_counts.min()}")

    def test_06_no_unexplained_data_loss(self):
        """Criterion 6: Total records in Train + Test equals total clean raw ratings."""
        self.assertEqual(len(self.train_df) + len(self.test_df), 100000)
        self.assertEqual(len(self.train_df), 80000)
        self.assertEqual(len(self.test_df), 20000)

    def test_07_sparse_matrix_shape_and_properties(self):
        """Criterion 7: Sparse User-Item matrix has correct dimensions and non-zero counts."""
        with open(PROCESSED_DIR / "user_to_idx.json", "r") as f:
            user_to_idx = {int(k): v for k, v in json.load(f).items()}
        with open(PROCESSED_DIR / "movie_to_idx.json", "r") as f:
            movie_to_idx = {int(k): v for k, v in json.load(f).items()}

        sparse_matrix = create_sparse_user_item_matrix(self.train_df, user_to_idx, movie_to_idx)

        self.assertIsInstance(sparse_matrix, csr_matrix)
        self.assertEqual(sparse_matrix.shape, (943, 1682))
        self.assertEqual(sparse_matrix.nnz, len(self.train_df))
        self.assertGreaterEqual(float(sparse_matrix.data.min()), 1.0)
        self.assertLessEqual(float(sparse_matrix.data.max()), 5.0)

    def test_08_processed_output_files_readable(self):
        """Criterion 8: All processed artifacts are valid, non-empty, and loadable."""
        required_files = [
            "train.csv",
            "test.csv",
            "movies_clean.csv",
            "split_metadata.json",
            "user_to_idx.json",
            "movie_to_idx.json",
        ]
        for filename in required_files:
            p = PROCESSED_DIR / filename
            self.assertTrue(p.is_file(), f"Missing required file: {p}")
            self.assertGreater(p.stat().st_size, 0, f"Empty file: {p}")

        with open(PROCESSED_DIR / "split_metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
            self.assertEqual(meta["random_state"], 42)
            self.assertEqual(meta["train_ratings"], 80000)
            self.assertEqual(meta["test_ratings"], 20000)

    def test_09_split_is_reproducible(self):
        """Criterion 9: Running split with seed 42 produces identical sets."""
        clean_r = clean_ratings(self.raw_ratings)
        train_reproduced, test_reproduced, _ = split_train_test_per_user(clean_r, test_ratio=0.2, random_state=42)

        pd.testing.assert_frame_equal(self.train_df, train_reproduced)
        pd.testing.assert_frame_equal(self.test_df, test_reproduced)

    def test_10_no_data_leakage(self):
        """Criterion 10: Strict absence of data leakage from test into train."""
        train_keys = self.train_df["user_id"].astype(str) + "_" + self.train_df["movie_id"].astype(str)
        test_keys = self.test_df["user_id"].astype(str) + "_" + self.test_df["movie_id"].astype(str)
        self.assertEqual(len(set(train_keys).intersection(set(test_keys))), 0)

        # Relevance threshold correctly marked
        self.assertTrue((self.train_df["is_relevant"] == (self.train_df["rating"] >= RELEVANT_RATING_THRESHOLD).astype(int)).all())
        self.assertTrue((self.test_df["is_relevant"] == (self.test_df["rating"] >= RELEVANT_RATING_THRESHOLD).astype(int)).all())


if __name__ == "__main__":
    unittest.main()
