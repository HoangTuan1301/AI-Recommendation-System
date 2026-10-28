"""MusicAI Recommendation Service Layer.

Orchestrates input validation, catalog metadata enrichment, user history tracking,
cold-start fallback handling, and real-time model inference.
"""

from pathlib import Path
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from backend.app.core.config import settings
from backend.app.schemas.music import (
    MusicHealthResponse,
    MusicModelInfo,
    MusicModelsListResponse,
    MusicRecommendationResponse,
    RecommendedSong,
    SongDetailResponse,
    UserHistoryResponse,
    UserHistorySong,
)
from src.music.models.registry import (
    MusicModelRegistry,
    MusicModelError,
    ModelNotFoundError,
    CheckpointError,
    music_model_registry,
)

logger = logging.getLogger(__name__)


class MusicRecommendationService:
    """Service mediating between REST API controllers and MusicAI model engines."""

    def __init__(
        self,
        registry: Optional[MusicModelRegistry] = None,
        data_dir: Optional[Path] = None,
    ) -> None:
        self.registry = registry or music_model_registry
        self.data_dir = Path(data_dir) if data_dir else settings.DATA_MUSIC_PATH

        self.songs_by_id: Dict[str, Dict[str, Any]] = {}
        self.all_catalog_songs: List[str] = []
        self.user_train_history: Dict[str, Set[str]] = {}
        self.user_train_records: Dict[str, List[Dict[str, Any]]] = {}
        self.song_stats: Dict[str, Dict[str, int]] = {}
        self.total_interactions: int = 0

        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load song catalog metadata and training interaction histories."""
        songs_file = self.data_dir / "songs.csv"
        train_file = self.data_dir / "train.csv"

        if not songs_file.exists():
            logger.warning(f"Songs metadata file not found at: {songs_file}")
            return

        # 1. Load songs catalog
        logger.info(f"Loading song catalog from {songs_file}...")
        df_songs = pd.read_csv(songs_file)
        for row in df_songs.itertuples(index=False):
            sid = str(row.song_id).strip()
            self.songs_by_id[sid] = {
                "song_id": sid,
                "title": getattr(row, "title", "Unknown Title"),
                "artist_name": getattr(row, "artist_name", "Unknown Artist"),
                "track_id": getattr(row, "track_id", None),
            }
        self.all_catalog_songs = sorted(self.songs_by_id.keys())
        logger.info(f"Loaded {len(self.songs_by_id):,} catalog songs.")

        # 2. Load training interactions
        if not train_file.exists():
            logger.warning(f"Training interactions file not found at: {train_file}")
            return

        logger.info(f"Loading user training history from {train_file}...")
        df_train = pd.read_csv(train_file)
        self.total_interactions = len(df_train)

        # Precompute user seen sets and detailed records
        user_seen: Dict[str, Set[str]] = {}
        user_records: Dict[str, List[Dict[str, Any]]] = {}
        song_plays: Dict[str, int] = {}
        song_listeners: Dict[str, Set[str]] = {}

        for row in df_train.itertuples(index=False):
            uid = str(row.user_id).strip()
            sid = str(row.song_id).strip()
            cnt = int(row.play_count)

            if uid not in user_seen:
                user_seen[uid] = set()
                user_records[uid] = []
            user_seen[uid].add(sid)
            user_records[uid].append({"song_id": sid, "play_count": cnt})

            # Song statistics
            song_plays[sid] = song_plays.get(sid, 0) + cnt
            if sid not in song_listeners:
                song_listeners[sid] = set()
            song_listeners[sid].add(uid)

        self.user_train_history = user_seen
        self.user_train_records = user_records

        self.song_stats = {
            sid: {
                "total_plays": song_plays.get(sid, 0),
                "unique_listeners": len(song_listeners.get(sid, set())),
            }
            for sid in self.all_catalog_songs
        }
        logger.info(
            f"Indexed {len(self.user_train_history):,} users and {self.total_interactions:,} train interactions."
        )

    def list_models(self) -> MusicModelsListResponse:
        """List all 6 MusicAI models and their operational readiness."""
        raw_models = self.registry.list_models()
        model_infos = [MusicModelInfo(**m) for m in raw_models]
        return MusicModelsListResponse(
            total=len(model_infos),
            default_model=self.registry.default_model_id,
            models=model_infos,
        )

    def get_recommendations(
        self,
        user_id: str,
        model_name: Optional[str] = None,
        top_k: int = 10,
    ) -> MusicRecommendationResponse:
        """Generate personalized song recommendations for a user.

        Args:
            user_id: Target user identifier string.
            model_name: Identifier of requested AI model. Defaults to 'item_cf'.
            top_k: Number of recommendations to return (1 <= top_k <= 100).

        Returns:
            MusicRecommendationResponse: Enriched recommendations with song metadata.
        """
        # 1. Input validation
        clean_user_id = str(user_id).strip()
        if not clean_user_id:
            raise ValueError("Parameter 'user_id' must be a non-empty string.")

        if top_k < 1 or top_k > 100:
            raise ValueError(f"Parameter 'top_k' must be between 1 and 100. Received: {top_k}")

        requested_model = (model_name or self.registry.default_model_id).strip().lower()
        if requested_model not in self.registry._definitions:
            valid_models = list(self.registry._definitions.keys())
            raise ModelNotFoundError(
                f"Invalid model '{requested_model}'. Available models: {valid_models}"
            )

        t_start = time.perf_counter()

        # 2. Check if user is known or completely new (cold-start)
        is_known_user = clean_user_id in self.user_train_history
        fallback_used = False
        fallback_reason: Optional[str] = None
        model_to_use = requested_model

        if not is_known_user:
            # Cold-start fallback: Popularity model
            model_to_use = "popularity"
            fallback_used = True
            fallback_reason = (
                f"User '{clean_user_id}' has no interaction history in training data. "
                f"Fell back to non-personalized Popularity Baseline."
            )
            logger.info(f"Cold-start fallback triggered for user '{clean_user_id}'.")

        # 3. Model execution with graceful degradation if checkpoint fails
        try:
            model = self.registry.get_model(model_to_use)
            raw_recs = model.recommend(
                user_id=clean_user_id,
                top_k=top_k,
                exclude_seen=True,
                candidate_songs=self.all_catalog_songs,
            )
        except CheckpointError as ckpt_err:
            logger.error(f"Checkpoint error for model '{model_to_use}': {ckpt_err}")
            # If requested model checkpoint failed, attempt popularity fallback
            if model_to_use != "popularity":
                logger.warning("Attempting emergency fallback to Popularity Baseline...")
                try:
                    pop_model = self.registry.get_model("popularity")
                    raw_recs = pop_model.recommend(
                        user_id=clean_user_id,
                        top_k=top_k,
                        exclude_seen=True,
                        candidate_songs=self.all_catalog_songs,
                    )
                    model_to_use = "popularity"
                    fallback_used = True
                    fallback_reason = f"Checkpoint for '{requested_model}' unavailable. Fallback to Popularity."
                except Exception as pop_exc:
                    raise CheckpointError(
                        f"Both requested model '{requested_model}' and fallback failed: {pop_exc}"
                    ) from pop_exc
            else:
                raise

        inference_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

        # 4. Format recommendations and enrich with metadata
        seen_train = self.user_train_history.get(clean_user_id, set())
        recommended_songs: List[RecommendedSong] = []
        seen_output_ids: Set[str] = set()

        for sid, raw_score in raw_recs:
            if sid in seen_train:
                # Double guard against seen items
                continue
            if sid in seen_output_ids:
                # Deduplication safeguard
                continue

            seen_output_ids.add(sid)
            meta = self.songs_by_id.get(sid, {})
            title = str(meta.get("title", f"Unknown Track ({sid})"))
            artist = str(meta.get("artist_name", "Unknown Artist"))
            track_id = meta.get("track_id")

            # Clean score: avoid NaN or Inf
            clean_score = 0.0
            if np.isfinite(raw_score):
                clean_score = round(float(raw_score), 4)

            recommended_songs.append(
                RecommendedSong(
                    rank=len(recommended_songs) + 1,
                    song_id=sid,
                    title=title,
                    artist_name=artist,
                    track_id=track_id,
                    score=clean_score,
                )
            )

            if len(recommended_songs) >= top_k:
                break

        return MusicRecommendationResponse(
            user_id=clean_user_id,
            model_requested=requested_model,
            model_used=model_to_use,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            top_k=top_k,
            count=len(recommended_songs),
            inference_time_ms=inference_ms,
            recommendations=recommended_songs,
        )

    def get_song(self, song_id: str) -> Optional[SongDetailResponse]:
        """Fetch metadata and listening statistics for an individual song."""
        clean_sid = str(song_id).strip()
        meta = self.songs_by_id.get(clean_sid)
        if not meta:
            return None

        stats_dict = self.song_stats.get(clean_sid, {})
        return SongDetailResponse(
            song_id=clean_sid,
            track_id=meta.get("track_id"),
            title=meta.get("title", "Unknown Title"),
            artist_name=meta.get("artist_name", "Unknown Artist"),
            total_play_count=stats_dict.get("total_plays", 0),
            unique_listeners=stats_dict.get("unique_listeners", 0),
        )

    def get_user_history(self, user_id: str, limit: int = 50) -> Optional[UserHistoryResponse]:
        """Fetch training listening history for a user.

        Guaranteed to only expose training split interactions, never test/validation.
        """
        clean_uid = str(user_id).strip()
        records = self.user_train_records.get(clean_uid)
        if records is None:
            return None

        sorted_records = sorted(records, key=lambda x: x["play_count"], reverse=True)[:limit]
        history_songs = []
        for r in sorted_records:
            sid = r["song_id"]
            meta = self.songs_by_id.get(sid, {})
            history_songs.append(
                UserHistorySong(
                    song_id=sid,
                    title=meta.get("title", f"Track {sid}"),
                    artist_name=meta.get("artist_name", "Unknown Artist"),
                    track_id=meta.get("track_id"),
                    play_count=r["play_count"],
                )
            )

        total_plays = sum(r["play_count"] for r in records)
        return UserHistoryResponse(
            user_id=clean_uid,
            total_plays=total_plays,
            total_songs=len(records),
            history=history_songs,
        )

    def get_health(self) -> MusicHealthResponse:
        """Return diagnostic health and domain statistics for MusicAI."""
        reg_status = self.registry.get_status()
        status_str = "healthy" if reg_status["available_checkpoints"] > 0 else "degraded"

        return MusicHealthResponse(
            status=status_str,
            service="MusicAI Song Recommendation Engine",
            domain="music",
            catalog_songs=len(self.songs_by_id),
            total_users=len(self.user_train_history),
            total_train_interactions=self.total_interactions,
            available_models=reg_status["available_checkpoints"],
            loaded_models=reg_status["loaded_model_ids"],
            default_model=self.registry.default_model_id,
        )


# Global singleton instance
music_service = MusicRecommendationService()
