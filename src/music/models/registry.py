"""Unified Model Registry for MusicAI Recommendation Models.

Manages catalog metadata, checkpoint lazy-loading, and real-time inference
for all 6 song-level recommendation models:
1. Popularity Baseline (popularity)
2. Item-Based Collaborative Filtering (item_cf) [Default]
3. User-Based Collaborative Filtering (user_cf)
4. Implicit Alternating Least Squares (ials)
5. Bayesian Personalized Ranking (bpr_mf)
6. Implicit Neural Collaborative Filtering (implicit_neural_cf)

Completely independent of CineAI's model registry to maintain domain isolation.
"""

from pathlib import Path
import logging
import threading
from typing import Any, Dict, List, Optional, Type
import joblib

from src.music.models.base_music_recommender import BaseMusicRecommender
from src.music.models.popularity import PopularityMusicRecommender
from src.music.models.item_cf import ItemCFMusicRecommender
from src.music.models.user_cf import UserCFMusicRecommender
from src.music.models.ials import IALSMusicRecommender
from src.music.models.bpr_mf import BPRMFMusicRecommender
from src.music.models.implicit_neural_cf import ImplicitNeuralCFRecommender

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_CHECKPOINTS_DIR = REPO_ROOT / "models" / "checkpoints" / "music"


class MusicModelError(Exception):
    """Base exception for MusicModelRegistry errors."""
    pass


class ModelNotFoundError(MusicModelError):
    """Raised when an unknown model_id is requested."""
    pass


class CheckpointError(MusicModelError):
    """Raised when a checkpoint file is missing or corrupted."""
    pass


# Official Registry Definitions for the 6 MusicAI models
MUSIC_MODEL_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "popularity": {
        "id": "popularity",
        "name": "Popularity Baseline",
        "family": "Heuristic / Baseline",
        "class": PopularityMusicRecommender,
        "checkpoint_file": "popularity.pkl",
        "is_personalized": False,
        "is_default": False,
        "description": "Gợi ý các bài hát có tổng lượt nghe nhiều nhất trên toàn hệ thống (không cá nhân hóa, dùng làm fallback).",
        "latency_profile": "< 1.0 ms",
    },
    "item_cf": {
        "id": "item_cf",
        "name": "Item-Based Collaborative Filtering",
        "family": "Neighborhood Collaborative Filtering",
        "class": ItemCFMusicRecommender,
        "checkpoint_file": "item_cf.pkl",
        "is_personalized": True,
        "is_default": True,
        "description": "Gợi ý dựa trên ma trận đồng nghe Cosine Top-50 láng giềng. Đạt độ chính xác và độ phủ danh mục cao nhất.",
        "latency_profile": "< 10.0 ms",
    },
    "user_cf": {
        "id": "user_cf",
        "name": "User-Based Collaborative Filtering",
        "family": "Neighborhood Collaborative Filtering",
        "class": UserCFMusicRecommender,
        "checkpoint_file": "user_cf.pkl",
        "is_personalized": True,
        "is_default": False,
        "description": "Tìm kiếm người dùng có gu âm nhạc tương đồng qua hệ số Cosine và tổng hợp các bài hát họ yêu thích.",
        "latency_profile": "< 2.0 ms",
    },
    "ials": {
        "id": "ials",
        "name": "Implicit Alternating Least Squares (iALS)",
        "family": "Matrix Factorization",
        "class": IALSMusicRecommender,
        "checkpoint_file": "ials.pkl",
        "is_personalized": True,
        "is_default": False,
        "description": "Phân rã ma trận tương tác ngầm định có trọng số tin cậy logarit sang không gian nhúng 32 chiều.",
        "latency_profile": "< 5.0 ms",
    },
    "bpr_mf": {
        "id": "bpr_mf",
        "name": "Bayesian Personalized Ranking (BPR-MF)",
        "family": "Pairwise Ranking",
        "class": BPRMFMusicRecommender,
        "checkpoint_file": "bpr_mf.pkl",
        "is_personalized": True,
        "is_default": False,
        "description": "Tối ưu hóa thứ hạng tương đối giữa bài hát đã nghe và chưa nghe qua hàm mất mát sigmoid BPR.",
        "latency_profile": "< 6.0 ms",
    },
    "implicit_neural_cf": {
        "id": "implicit_neural_cf",
        "name": "Implicit Neural Collaborative Filtering (NeuMF)",
        "family": "Deep Learning Hybrid",
        "class": ImplicitNeuralCFRecommender,
        "checkpoint_file": "implicit_neumf.pkl",
        "is_personalized": True,
        "is_default": False,
        "description": "Mô hình học sâu kết hợp Generalized Matrix Factorization (GMF) và Multi-Layer Perceptron (MLP) phi tuyến.",
        "latency_profile": "< 4.0 ms",
    },
}


class MusicModelRegistry:
    """Thread-safe registry managing checkpoint lifecycle, metadata and inference caching."""

    def __init__(self, checkpoints_dir: Optional[Path] = None) -> None:
        self.checkpoints_dir = Path(checkpoints_dir) if checkpoints_dir else DEFAULT_CHECKPOINTS_DIR
        self._definitions = MUSIC_MODEL_DEFINITIONS
        self._loaded_models: Dict[str, BaseMusicRecommender] = {}
        self._lock = threading.Lock()
        self.default_model_id: str = "item_cf"

    def list_models(self) -> List[Dict[str, Any]]:
        """Return status and metadata for all 6 models."""
        models_list = []
        for model_id, defn in self._definitions.items():
            ckpt_path = self.checkpoints_dir / defn["checkpoint_file"]
            exists = ckpt_path.is_file() and ckpt_path.stat().st_size > 0
            size_kb = round(ckpt_path.stat().st_size / 1024.0, 1) if exists else None

            models_list.append({
                "id": model_id,
                "name": defn["name"],
                "family": defn["family"],
                "description": defn["description"],
                "is_personalized": defn["is_personalized"],
                "is_default": (model_id == self.default_model_id),
                "is_available": exists,
                "is_loaded": (model_id in self._loaded_models),
                "checkpoint_file": defn["checkpoint_file"],
                "checkpoint_size_kb": size_kb,
                "latency_profile": defn["latency_profile"],
            })
        return models_list

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve model metadata by model_id."""
        defn = self._definitions.get(model_id)
        if not defn:
            return None
        ckpt_path = self.checkpoints_dir / defn["checkpoint_file"]
        exists = ckpt_path.is_file() and ckpt_path.stat().st_size > 0
        size_kb = round(ckpt_path.stat().st_size / 1024.0, 1) if exists else None

        return {
            "id": model_id,
            "name": defn["name"],
            "family": defn["family"],
            "description": defn["description"],
            "is_personalized": defn["is_personalized"],
            "is_default": (model_id == self.default_model_id),
            "is_available": exists,
            "is_loaded": (model_id in self._loaded_models),
            "checkpoint_file": defn["checkpoint_file"],
            "checkpoint_size_kb": size_kb,
            "latency_profile": defn["latency_profile"],
        }

    def is_available(self, model_id: str) -> bool:
        """Check if model checkpoint exists on disk."""
        defn = self._definitions.get(model_id)
        if not defn:
            return False
        ckpt_path = self.checkpoints_dir / defn["checkpoint_file"]
        return ckpt_path.is_file() and ckpt_path.stat().st_size > 0

    def get_model(self, model_id: str) -> BaseMusicRecommender:
        """Fetch model instance from memory cache or lazy-load from disk checkpoint.

        Thread-safe to prevent race conditions during concurrent cold starts.
        Never triggers online training.
        """
        if model_id not in self._definitions:
            valid_models = list(self._definitions.keys())
            raise ModelNotFoundError(
                f"Model '{model_id}' is not registered. Available models: {valid_models}"
            )

        # Fast path without lock
        if model_id in self._loaded_models:
            return self._loaded_models[model_id]

        with self._lock:
            # Double-check inside lock
            if model_id in self._loaded_models:
                return self._loaded_models[model_id]

            defn = self._definitions[model_id]
            ckpt_path = self.checkpoints_dir / defn["checkpoint_file"]

            if not ckpt_path.exists():
                raise CheckpointError(
                    f"Checkpoint for model '{model_id}' does not exist at '{ckpt_path}'."
                )

            if ckpt_path.stat().st_size == 0:
                raise CheckpointError(
                    f"Checkpoint file '{ckpt_path}' is corrupted (zero bytes)."
                )

            logger.info(f"Loading MusicAI model '{model_id}' from checkpoint: {ckpt_path}...")
            try:
                model_cls = defn["class"]
                loaded_model = model_cls.load(ckpt_path)
                self._loaded_models[model_id] = loaded_model
                logger.info(f"Successfully loaded model '{model_id}' into memory.")
                return loaded_model
            except Exception as exc:
                logger.error(f"Failed to load checkpoint '{ckpt_path}' for model '{model_id}': {exc}")
                raise CheckpointError(f"Error loading checkpoint for '{model_id}': {exc}") from exc

    def preload_all(self) -> Dict[str, bool]:
        """Eagerly load all available models into memory cache."""
        status = {}
        for model_id in self._definitions.keys():
            try:
                self.get_model(model_id)
                status[model_id] = True
            except Exception as e:
                logger.warning(f"Could not preload model '{model_id}': {e}")
                status[model_id] = False
        return status

    def unload(self, model_id: Optional[str] = None) -> None:
        """Evict model(s) from memory cache to free RAM."""
        with self._lock:
            if model_id:
                self._loaded_models.pop(model_id, None)
            else:
                self._loaded_models.clear()

    def get_status(self) -> Dict[str, Any]:
        """Return overall registry diagnostic status."""
        available_count = sum(1 for m in self._definitions if self.is_available(m))
        loaded_count = len(self._loaded_models)
        return {
            "total_registered": len(self._definitions),
            "available_checkpoints": available_count,
            "loaded_models_count": loaded_count,
            "loaded_model_ids": list(self._loaded_models.keys()),
            "default_model": self.default_model_id,
            "checkpoints_dir": str(self.checkpoints_dir),
        }


# Global Singleton for Music Model Registry
music_model_registry = MusicModelRegistry()
