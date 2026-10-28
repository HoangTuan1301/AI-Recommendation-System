"""MSD Taste Profile Data Acquisition, Cleaning, Sampling, and Splitting Pipeline.

This script implements a reproducible, memory-efficient data engineering pipeline for
the Million Song Dataset (MSD) Taste Profile Subset and MSD Metadata.
It builds a high-quality, song-level implicit recommendation dataset:
- Filters active users and connected songs with minimum interaction thresholds.
- Resolves song-to-track metadata deterministically.
- Generates 80/10/10 user-stratified train/validation/test splits (seed=42).
- Produces dataset manifests, data contracts, and quality reports.
"""

from pathlib import Path
import sys
import os
import json
import hashlib
import zipfile
import logging
from typing import Dict, List, Optional, Set, Tuple, Any
import requests
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("MSD_Ingestion")

# Repository root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "music"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = REPO_ROOT / "reports" / "musicai"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Official Source URLs
TRIPLETS_URL = "http://labrosa.ee.columbia.edu/~dpwe/tmp/train_triplets.txt.zip"
METADATA_URL = "http://millionsongdataset.com/sites/default/files/AdditionalFiles/unique_tracks.txt"

TRIPLETS_ZIP_PATH = RAW_DIR / "train_triplets.txt.zip"
METADATA_TXT_PATH = RAW_DIR / "unique_tracks.txt"

RANDOM_SEED = 42
TARGET_USERS = 3000
MIN_SONGS_PER_USER = 15
MAX_SONGS_PER_USER = 150
MIN_LISTENERS_PER_SONG = 8


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest_path: Path, chunk_size: int = 1024 * 1024) -> None:
    """Download a file with progress logging, temp file, and resume/skip check."""
    if dest_path.exists() and dest_path.stat().st_size > 0:
        logger.info(f"File already exists: {dest_path} ({dest_path.stat().st_size / (1024*1024):.2f} MB). Skipping download.")
        return

    logger.info(f"Initiating download from: {url}")
    temp_path = dest_path.with_suffix(".tmp")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        header_len = r.headers.get("content-length")
        total_size = int(header_len) if header_len is not None and header_len.isdigit() else 0
        downloaded = 0
        with open(temp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and downloaded % (20 * 1024 * 1024) < chunk_size:
                        pct = (downloaded / total_size) * 100
                        logger.info(f"Downloaded {downloaded / (1024*1024):.1f} / {total_size / (1024*1024):.1f} MB ({pct:.1f}%)")
                    elif total_size == 0 and downloaded % (10 * 1024 * 1024) < chunk_size:
                        logger.info(f"Downloaded {downloaded / (1024*1024):.1f} MB (streaming chunked transfer)...")

    temp_path.rename(dest_path)
    logger.info(f"Successfully downloaded to: {dest_path} ({dest_path.stat().st_size / (1024*1024):.2f} MB)")


def load_and_filter_metadata(valid_song_ids: Optional[Set[str]] = None) -> pd.DataFrame:
    """Load unique_tracks.txt and extract song metadata deterministically.

    Line format: track_id<SEP>song_id<SEP>artist_name<SEP>song_title
    """
    logger.info(f"Loading song metadata from {METADATA_TXT_PATH}...")
    metadata_rows = []
    seen_song_ids = set()

    with open(METADATA_TXT_PATH, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split("<SEP>")
            if len(parts) < 4:
                continue
            track_id, song_id, artist_name, song_title = parts[0], parts[1], parts[2], parts[3]
            
            # If valid_song_ids filter is provided, skip unmatched songs
            if valid_song_ids is not None and song_id not in valid_song_ids:
                continue

            # Deterministic resolution: keep first appearance of song_id
            if song_id not in seen_song_ids:
                seen_song_ids.add(song_id)
                metadata_rows.append({
                    "song_id": song_id,
                    "title": song_title if song_title else "Unknown Title",
                    "artist_name": artist_name if artist_name else "Unknown Artist",
                    "track_id": track_id,
                })

    df_meta = pd.DataFrame(metadata_rows)
    logger.info(f"Metadata extracted: {len(df_meta):,} unique songs.")
    return df_meta


def sample_and_clean_triplets(
    zip_path: Path,
    target_users: int = TARGET_USERS,
    min_user_songs: int = MIN_SONGS_PER_USER,
    max_user_songs: int = MAX_SONGS_PER_USER,
    min_song_listeners: int = MIN_LISTENERS_PER_SONG,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Stream and sample reproducible core subset from train_triplets.txt.zip.

    Line format: user_id\tsong_id\tplay_count
    """
    logger.info("Opening train_triplets.txt.zip for streaming extraction...")
    np.random.seed(random_seed)

    # Pass 1: Collect user interaction counts to select valid active users
    user_counts: Dict[str, int] = {}
    logger.info("Pass 1: Scanning user activity distribution from zip stream...")

    with zipfile.ZipFile(zip_path, "r") as z:
        # Find the text file inside the zip
        txt_filename = [n for n in z.namelist() if n.endswith(".txt")][0]
        with z.open(txt_filename, "r") as f:
            for idx, line in enumerate(f):
                parts = line.decode("utf-8", errors="replace").strip().split("\t")
                if len(parts) == 3:
                    u_id = parts[0]
                    user_counts[u_id] = user_counts.get(u_id, 0) + 1
                
                if idx > 0 and idx % 5_000_000 == 0:
                    logger.info(f"Pass 1: Processed {idx:,} lines... Unique users seen: {len(user_counts):,}")

    logger.info(f"Pass 1 complete. Total unique users scanned: {len(user_counts):,}")

    # Filter eligible users within activity window [min_user_songs, max_user_songs]
    eligible_users = [
        u for u, count in user_counts.items()
        if min_user_songs <= count <= max_user_songs
    ]
    logger.info(f"Eligible users with [{min_user_songs}, {max_user_songs}] interactions: {len(eligible_users):,}")

    # Deterministically sample target_users
    rng = np.random.default_rng(random_seed)
    selected_users = set(rng.choice(eligible_users, size=min(target_users, len(eligible_users)), replace=False))
    logger.info(f"Sampled {len(selected_users):,} target users (seed={random_seed}).")

    # Pass 2: Extract interactions for selected users
    logger.info("Pass 2: Extracting triplets for selected users...")
    sampled_triplets = []
    with zipfile.ZipFile(zip_path, "r") as z:
        txt_filename = [n for n in z.namelist() if n.endswith(".txt")][0]
        with z.open(txt_filename, "r") as f:
            for line in f:
                parts = line.decode("utf-8", errors="replace").strip().split("\t")
                if len(parts) == 3:
                    u_id, s_id, count_str = parts[0], parts[1], parts[2]
                    if u_id in selected_users:
                        try:
                            play_count = int(count_str)
                            if play_count > 0:
                                sampled_triplets.append((u_id, s_id, play_count))
                        except ValueError:
                            continue

    df_interactions = pd.DataFrame(sampled_triplets, columns=["user_id", "song_id", "play_count"])
    logger.info(f"Pass 2 extracted {len(df_interactions):,} raw interactions.")

    # Deduplicate user-song pairs (aggregate with max play_count if duplicates exist)
    df_interactions = (
        df_interactions.groupby(["user_id", "song_id"], as_index=False)["play_count"]
        .max()
    )

    # Filter songs with minimum listener count
    song_listener_counts = df_interactions.groupby("song_id")["user_id"].nunique()
    valid_songs = set(song_listener_counts[song_listener_counts >= min_song_listeners].index)
    df_clean = df_interactions[df_interactions["song_id"].isin(valid_songs)].copy()

    # Re-filter users to ensure they still have >= 5 interactions after song filtering
    user_interaction_counts = df_clean.groupby("user_id")["song_id"].nunique()
    active_users = set(user_interaction_counts[user_interaction_counts >= 5].index)
    df_clean = df_clean[df_clean["user_id"].isin(active_users)].copy()

    logger.info(
        f"Cleaning complete: {len(df_clean):,} interactions across "
        f"{df_clean['user_id'].nunique():,} users and {df_clean['song_id'].nunique():,} songs."
    )
    return df_clean


def split_user_stratified(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split interactions per user into Train (80%), Val (10%), Test (10%).

    Ensures warm-start constraint: Every song in Val and Test must appear in Train.
    """
    logger.info(f"Performing user-stratified holdout split ({train_ratio:.0%}/{val_ratio:.0%}/{test_ratio:.0%})...")
    rng = np.random.default_rng(random_seed)

    train_rows, val_rows, test_rows = [], [], []

    for user_id, group in df.groupby("user_id"):
        n_items = len(group)
        if n_items < 5:
            # All to train for sparse users
            train_rows.append(group)
            continue

        shuffled_indices = rng.permutation(group.index.to_numpy())
        n_test = max(1, int(round(n_items * test_ratio)))
        n_val = max(1, int(round(n_items * val_ratio)))
        
        # Ensure at least 3 items remain in train
        if n_items - (n_val + n_test) < 3:
            n_test = 1
            n_val = 1

        test_idx = shuffled_indices[:n_test]
        val_idx = shuffled_indices[n_test : n_test + n_val]
        train_idx = shuffled_indices[n_test + n_val :]

        test_rows.append(df.loc[test_idx])
        val_rows.append(df.loc[val_idx])
        train_rows.append(df.loc[train_idx])

    train_df = pd.concat(train_rows, ignore_index=True)
    val_df = pd.concat(val_rows, ignore_index=True)
    test_df = pd.concat(test_rows, ignore_index=True)

    # Warm-start catalog enforcement: move any val/test item not in train back to train
    train_songs = set(train_df["song_id"])

    val_cold = val_df[~val_df["song_id"].isin(train_songs)]
    if len(val_cold) > 0:
        logger.info(f"Moving {len(val_cold)} cold-start validation interactions to train...")
        train_df = pd.concat([train_df, val_cold], ignore_index=True)
        val_df = val_df[val_df["song_id"].isin(train_songs)].copy()

    test_cold = test_df[~test_df["song_id"].isin(train_songs)]
    if len(test_cold) > 0:
        logger.info(f"Moving {len(test_cold)} cold-start test interactions to train...")
        train_df = pd.concat([train_df, test_cold], ignore_index=True)
        test_df = test_df[test_df["song_id"].isin(train_songs)].copy()

    logger.info(
        f"Splits generated: Train={len(train_df):,} ({len(train_df)/len(df):.1%}), "
        f"Val={len(val_df):,} ({len(val_df)/len(df):.1%}), "
        f"Test={len(test_df):,} ({len(test_df)/len(df):.1%})"
    )
    return train_df, val_df, test_df


def generate_dataset_manifest_and_mappings(
    interactions_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    songs_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Create index mappings and export dataset manifest JSON."""
    logger.info("Generating index mappings and manifest...")

    # User mapping
    unique_users = sorted(interactions_df["user_id"].unique())
    user_mapping_df = pd.DataFrame({
        "raw_user_id": unique_users,
        "user_idx": range(len(unique_users)),
    })
    user_mapping_df.to_csv(PROCESSED_DIR / "user_mapping.csv", index=False)

    # Song mapping
    unique_songs = sorted(interactions_df["song_id"].unique())
    song_mapping_df = pd.DataFrame({
        "raw_song_id": unique_songs,
        "song_idx": range(len(unique_songs)),
    })
    song_mapping_df.to_csv(PROCESSED_DIR / "song_mapping.csv", index=False)

    # Save datasets
    interactions_df.to_csv(PROCESSED_DIR / "interactions.csv", index=False)
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "validation.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)
    songs_df.to_csv(PROCESSED_DIR / "songs.csv", index=False)

    # Statistics
    n_users = len(unique_users)
    n_songs = len(unique_songs)
    n_interactions = len(interactions_df)
    sparsity = 1.0 - (n_interactions / (n_users * n_songs))
    density = 100.0 - (sparsity * 100.0)

    manifest = {
        "dataset_name": "MSD_Taste_Profile_Song_Level_Subset",
        "version": "1.0.0",
        "created_date": "2026-10-28",
        "random_seed": RANDOM_SEED,
        "source_triplets": TRIPLETS_URL,
        "source_metadata": METADATA_URL,
        "counts": {
            "users": n_users,
            "songs": n_songs,
            "total_interactions": n_interactions,
            "train_interactions": len(train_df),
            "val_interactions": len(val_df),
            "test_interactions": len(test_df),
        },
        "sparsity_metrics": {
            "sparsity_pct": round(sparsity * 100, 4),
            "density_pct": round(density, 4),
            "mean_songs_per_user": round(float(interactions_df.groupby("user_id")["song_id"].count().mean()), 2),
            "median_songs_per_user": int(interactions_df.groupby("user_id")["song_id"].count().median()),
            "mean_users_per_song": round(float(interactions_df.groupby("song_id")["user_id"].count().mean()), 2),
            "median_users_per_song": int(interactions_df.groupby("song_id")["user_id"].count().median()),
        },
        "play_count_distribution": {
            "min": int(interactions_df["play_count"].min()),
            "p25": float(interactions_df["play_count"].quantile(0.25)),
            "median": float(interactions_df["play_count"].median()),
            "p75": float(interactions_df["play_count"].quantile(0.75)),
            "p95": float(interactions_df["play_count"].quantile(0.95)),
            "max": int(interactions_df["play_count"].max()),
        },
        "metadata_coverage": {
            "songs_with_metadata": len(songs_df),
            "coverage_pct": 100.0,
        },
        "checksums": {
            "interactions_csv": compute_file_sha256(PROCESSED_DIR / "interactions.csv"),
            "train_csv": compute_file_sha256(PROCESSED_DIR / "train.csv"),
            "validation_csv": compute_file_sha256(PROCESSED_DIR / "validation.csv"),
            "test_csv": compute_file_sha256(PROCESSED_DIR / "test.csv"),
            "songs_csv": compute_file_sha256(PROCESSED_DIR / "songs.csv"),
        },
    }

    manifest_path = PROCESSED_DIR / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest written to: {manifest_path}")
    return manifest


def run_pipeline():
    """Execute end-to-end data engineering pipeline."""
    logger.info("=== STARTING MSD TASTE PROFILE INGESTION PIPELINE ===")

    # Step 1: Download raw source files
    logger.info("Step 1: Downloading triplets and metadata...")
    download_file(TRIPLETS_URL, TRIPLETS_ZIP_PATH)
    download_file(METADATA_URL, METADATA_TXT_PATH)

    # Step 2: Sample and clean triplets from zip stream
    logger.info("Step 2: Sampling and cleaning triplets...")
    df_interactions = sample_and_clean_triplets(
        TRIPLETS_ZIP_PATH,
        target_users=TARGET_USERS,
        min_user_songs=MIN_SONGS_PER_USER,
        max_user_songs=MAX_SONGS_PER_USER,
        min_song_listeners=MIN_LISTENERS_PER_SONG,
        random_seed=RANDOM_SEED,
    )

    # Step 3: Match and filter metadata
    logger.info("Step 3: Resolving song metadata...")
    valid_song_ids = set(df_interactions["song_id"].unique())
    df_songs = load_and_filter_metadata(valid_song_ids)

    # Enforce 100% metadata join
    matched_song_ids = set(df_songs["song_id"])
    initial_count = len(df_interactions)
    df_interactions = df_interactions[df_interactions["song_id"].isin(matched_song_ids)].copy()
    dropped = initial_count - len(df_interactions)
    if dropped > 0:
        logger.info(f"Dropped {dropped} interactions without metadata matches.")

    # Step 4: Split into Train, Validation, Test
    logger.info("Step 4: Performing user-stratified holdout split...")
    train_df, val_df, test_df = split_user_stratified(
        df_interactions,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_seed=RANDOM_SEED,
    )

    # Step 5: Export dataset, mappings, and manifest
    logger.info("Step 5: Exporting processed datasets and manifest...")
    manifest = generate_dataset_manifest_and_mappings(
        df_interactions, train_df, val_df, test_df, df_songs
    )

    # Step 6: Generate EDA and quality reports
    logger.info("Step 6: Generating EDA and Data Quality reports...")
    try:
        from scripts.music.generate_music_eda_reports import generate_reports
        generate_reports()
    except Exception as e:
        logger.warning(f"Report generation encountered error: {e}")

    logger.info("=== PIPELINE EXECUTION COMPLETE ===")
    return manifest


if __name__ == "__main__":
    run_pipeline()
