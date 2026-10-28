"""Unit Tests for MusicModelRegistry."""

from pathlib import Path
import pytest

from src.music.models.registry import (
    MusicModelRegistry,
    ModelNotFoundError,
    CheckpointError,
    music_model_registry,
    MUSIC_MODEL_DEFINITIONS,
)
from src.music.models.base_music_recommender import BaseMusicRecommender


class TestMusicModelRegistry:
    """Test suite covering lifecycle, loading, and error handling of MusicModelRegistry."""

    def test_01_registry_definitions_completeness(self):
        """Verify all 6 official models are declared with required metadata."""
        expected_ids = {"popularity", "item_cf", "user_cf", "ials", "bpr_mf", "implicit_neural_cf"}
        actual_ids = set(music_model_registry._definitions.keys())
        assert expected_ids == actual_ids, f"Mismatch in model definitions: {expected_ids ^ actual_ids}"

        # Verify item_cf is declared as default
        assert music_model_registry.default_model_id == "item_cf"
        assert music_model_registry._definitions["item_cf"]["is_default"] is True

    def test_02_list_models_status(self):
        """Verify list_models() returns metadata for all 6 models with availability."""
        models = music_model_registry.list_models()
        assert len(models) == 6
        for m in models:
            assert "id" in m
            assert "name" in m
            assert "family" in m
            assert "is_available" in m
            assert "is_loaded" in m
            assert "checkpoint_size_kb" in m
            # All 6 checkpoints should exist from Phase M2
            assert m["is_available"] is True, f"Model {m['id']} checkpoint should exist."

    def test_03_lazy_load_item_cf(self):
        """Verify loading the default item_cf model returns a fitted BaseMusicRecommender."""
        model = music_model_registry.get_model("item_cf")
        assert isinstance(model, BaseMusicRecommender)
        assert model.is_fitted is True

        # Calling again should return cached instance (same memory address)
        model_cached = music_model_registry.get_model("item_cf")
        assert model is model_cached

    def test_04_lazy_load_all_models(self):
        """Verify all 6 models load successfully from checkpoints without training."""
        for m_id in ["popularity", "user_cf", "ials", "bpr_mf", "implicit_neural_cf"]:
            model = music_model_registry.get_model(m_id)
            assert isinstance(model, BaseMusicRecommender)
            assert model.is_fitted is True

    def test_05_unknown_model_raises_model_not_found(self):
        """Verify requesting an unregistered model raises ModelNotFoundError."""
        with pytest.raises(ModelNotFoundError) as exc_info:
            music_model_registry.get_model("non_existent_super_model")
        assert "is not registered" in str(exc_info.value)

    def test_06_missing_checkpoint_raises_checkpoint_error(self, tmp_path):
        """Verify pointing to an empty checkpoints directory raises CheckpointError."""
        empty_registry = MusicModelRegistry(checkpoints_dir=tmp_path)
        with pytest.raises(CheckpointError) as exc_info:
            empty_registry.get_model("item_cf")
        assert "does not exist" in str(exc_info.value)

    def test_07_status_diagnostics(self):
        """Verify get_status() returns accurate operational statistics."""
        status = music_model_registry.get_status()
        assert status["total_registered"] == 6
        assert status["available_checkpoints"] == 6
        assert status["default_model"] == "item_cf"
        assert len(status["loaded_model_ids"]) >= 1
