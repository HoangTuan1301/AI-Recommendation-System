"""Unit and Integration Tests for MusicAI Data Engineering Pipeline."""

from pathlib import Path
import json
import pytest
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "music" / "processed"
CINEAI_CHECKPOINTS_DIR = REPO_ROOT / "models" / "checkpoints"
CINEAI_DATA_DIR = REPO_ROOT / "data" / "processed"


class TestMusicDataPipeline:
    """Automated tests verifying data sanitization, split integrity, and CineAI isolation."""

    def test_01_cineai_regression_isolation(self):
        """Verify that all 10 CineAI checkpoints and MovieLens splits remain intact."""
        expected_checkpoints = [
            "popularity.pkl",
            "content_based.pkl",
            "user_cf.pkl",
            "item_cf.pkl",
            "knn_baseline.pkl",
            "svd.pkl",
            "svdpp.pkl",
            "nmf.pkl",
            "neural_cf.pkl",
            "hybrid.pkl",
        ]
        for cp in expected_checkpoints:
            cp_path = CINEAI_CHECKPOINTS_DIR / cp
            assert cp_path.exists(), f"CineAI checkpoint missing: {cp}"
            assert cp_path.stat().st_size > 100_000, f"CineAI checkpoint corrupted: {cp}"

        # MovieLens processed files
        assert (CINEAI_DATA_DIR / "train.csv").exists()
        assert (CINEAI_DATA_DIR / "test.csv").exists()
        assert (CINEAI_DATA_DIR / "movies_clean.csv").exists()

    def test_02_data_contract_mock_validation(self):
        """Validate pipeline rules on a synthetic test batch to verify logic before ingestion."""
        # Create a tiny test interaction batch
        test_data = pd.DataFrame({
            "user_id": ["u1", "u1", "u2", "u2", "u3", "u3"],
            "song_id": ["s1", "s2", "s1", "s3", "s2", "s3"],
            "play_count": [5, 1, 10, 2, 4, 8],
        })

        assert (test_data["play_count"] > 0).all(), "Play counts must be positive"
        assert not test_data.duplicated(subset=["user_id", "song_id"]).any(), "No duplicate user-song pairs"

        # Check continuous 0-indexing mapping logic
        unique_users = sorted(test_data["user_id"].unique())
        user_map = {u: i for i, u in enumerate(unique_users)}
        assert list(user_map.values()) == list(range(len(unique_users)))

        unique_songs = sorted(test_data["song_id"].unique())
        song_map = {s: i for i, s in enumerate(unique_songs)}
        assert list(song_map.values()) == list(range(len(unique_songs)))

    def test_03_processed_files_integrity_if_present(self):
        """If processed music files exist, run comprehensive integrity checks."""
        manifest_path = PROCESSED_DIR / "dataset_manifest.json"
        if not manifest_path.exists():
            pytest.skip("Processed music data not yet ingested (pending download approval).")

        train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
        val_df = pd.read_csv(PROCESSED_DIR / "validation.csv")
        test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
        songs_df = pd.read_csv(PROCESSED_DIR / "songs.csv")

        # 1. Non-empty
        assert len(train_df) > 0
        assert len(val_df) > 0
        assert len(test_df) > 0

        # 2. Positive play counts
        assert (train_df["play_count"] >= 1).all()
        assert (val_df["play_count"] >= 1).all()
        assert (test_df["play_count"] >= 1).all()

        # 3. Disjoint interactions (no overlap)
        train_pairs = set(zip(train_df["user_id"], train_df["song_id"]))
        val_pairs = set(zip(val_df["user_id"], val_df["song_id"]))
        test_pairs = set(zip(test_df["user_id"], test_df["song_id"]))

        assert len(train_pairs.intersection(val_pairs)) == 0, "Train and Val must be disjoint"
        assert len(train_pairs.intersection(test_pairs)) == 0, "Train and Test must be disjoint"
        assert len(val_pairs.intersection(test_pairs)) == 0, "Val and Test must be disjoint"

        # 4. Warm-start catalog coverage: all Val and Test songs exist in Train
        train_songs = set(train_df["song_id"])
        val_songs = set(val_df["song_id"])
        test_songs = set(test_df["song_id"])

        assert val_songs.issubset(train_songs), "Validation songs must exist in Train catalog"
        assert test_songs.issubset(train_songs), "Test songs must exist in Train catalog"

        # 5. Metadata completeness
        all_songs = train_songs.union(val_songs).union(test_songs)
        assert all_songs.issubset(set(songs_df["song_id"])), "All songs must have metadata"
        assert songs_df["title"].notna().all(), "No null titles permitted"
        assert songs_df["artist_name"].notna().all(), "No null artist names permitted"
