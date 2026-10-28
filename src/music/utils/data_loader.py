"""MusicAI Dataset Loader and Sparse Matrix Utilities.

Handles loading of processed CSV datasets, ID mapping conversions,
and sparse interaction matrix generation for implicit feedback models.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import json
import logging
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "music" / "processed"


class MusicDataLoader:
    """Manages processed MusicAI datasets, index mappings, and sparse matrices."""

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self.data_dir = data_dir or DATA_DIR
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Processed music data directory not found: {self.data_dir}")

        # 1. Load mappings
        self.user_map_df = pd.read_csv(self.data_dir / "user_mapping.csv")
        self.song_map_df = pd.read_csv(self.data_dir / "song_mapping.csv")

        self.user_to_idx: Dict[str, int] = dict(zip(self.user_map_df["raw_user_id"], self.user_map_df["user_idx"]))
        self.idx_to_user: Dict[int, str] = dict(zip(self.user_map_df["user_idx"], self.user_map_df["raw_user_id"]))

        self.song_to_idx: Dict[str, int] = dict(zip(self.song_map_df["raw_song_id"], self.song_map_df["song_idx"]))
        self.idx_to_song: Dict[int, str] = dict(zip(self.song_map_df["song_idx"], self.song_map_df["raw_song_id"]))

        self.num_users: int = len(self.user_to_idx)
        self.num_songs: int = len(self.song_to_idx)

        # 2. Load song metadata
        self.songs_df = pd.read_csv(self.data_dir / "songs.csv")
        self.song_metadata: Dict[str, Dict[str, str]] = {}
        for row in self.songs_df.itertuples(index=False):
            self.song_metadata[row.song_id] = {
                "title": getattr(row, "title", "Unknown Title"),
                "artist_name": getattr(row, "artist_name", "Unknown Artist"),
                "track_id": getattr(row, "track_id", ""),
            }

        # 3. Load interaction splits
        self.train_df = pd.read_csv(self.data_dir / "train.csv")
        self.val_df = pd.read_csv(self.data_dir / "validation.csv")
        self.test_df = pd.read_csv(self.data_dir / "test.csv")

        # 4. User seen history from train
        self.user_train_history: Dict[str, Set[str]] = (
            self.train_df.groupby("user_id")["song_id"].apply(set).to_dict()
        )
        # Ensure all users exist in history dict
        for u in self.user_to_idx.keys():
            if u not in self.user_train_history:
                self.user_train_history[u] = set()

        self.all_train_songs: List[str] = sorted(self.train_df["song_id"].unique())
        self.all_train_users: List[str] = sorted(self.train_df["user_id"].unique())

    def get_train_df(self) -> pd.DataFrame:
        return self.train_df.copy()

    def get_val_df(self) -> pd.DataFrame:
        return self.val_df.copy()

    def get_test_df(self) -> pd.DataFrame:
        return self.test_df.copy()

    def build_interaction_matrix(self, alpha: float = 40.0, use_log: bool = True) -> csr_matrix:
        """Construct user-song confidence CSR sparse matrix for training.

        Confidence formula: c_ui = 1 + alpha * log(1 + r_ui) if use_log else 1 + alpha * r_ui.
        Matrix shape: (num_users, num_songs).
        """
        user_indices = np.array([self.user_to_idx[u] for u in self.train_df["user_id"]], dtype=np.int32)
        song_indices = np.array([self.song_to_idx[s] for s in self.train_df["song_id"]], dtype=np.int32)
        play_counts = self.train_df["play_count"].to_numpy(dtype=np.float32)

        if use_log:
            confidences = 1.0 + alpha * np.log1p(play_counts)
        else:
            confidences = 1.0 + alpha * play_counts

        return csr_matrix(
            (confidences, (user_indices, song_indices)),
            shape=(self.num_users, self.num_songs),
            dtype=np.float32,
        )

    def build_binary_matrix(self) -> csr_matrix:
        """Construct binary interaction CSR sparse matrix: 1 if interacted else 0."""
        user_indices = np.array([self.user_to_idx[u] for u in self.train_df["user_id"]], dtype=np.int32)
        song_indices = np.array([self.song_to_idx[s] for s in self.train_df["song_id"]], dtype=np.int32)
        data = np.ones(len(self.train_df), dtype=np.float32)

        return csr_matrix(
            (data, (user_indices, song_indices)),
            shape=(self.num_users, self.num_songs),
            dtype=np.float32,
        )
