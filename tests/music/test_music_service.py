"""Unit Tests for MusicRecommendationService."""

import pytest
from backend.app.services.music_service import MusicRecommendationService
from src.music.models.registry import ModelNotFoundError


@pytest.fixture(scope="module")
def music_svc():
    """Shared MusicRecommendationService fixture."""
    return MusicRecommendationService()


class TestMusicRecommendationService:
    """Test suite covering recommendation orchestration, filtering, and catalog lookup."""

    def test_01_service_initialization(self, music_svc):
        """Verify catalog metadata and user histories are correctly indexed."""
        assert len(music_svc.songs_by_id) == 2706
        assert len(music_svc.user_train_history) == 2520
        assert music_svc.total_interactions == 36994
        assert len(music_svc.all_catalog_songs) == 2706

    def test_02_recommendations_known_user_item_cf(self, music_svc):
        """Verify recommendations for an active user using default Item-CF."""
        user_id = "0023d50b3c99c4352585408b1b21f3762655f434"
        res = music_svc.get_recommendations(user_id=user_id, model_name="item_cf", top_k=10)

        assert res.user_id == user_id
        assert res.model_requested == "item_cf"
        assert res.model_used == "item_cf"
        assert res.fallback_used is False
        assert res.fallback_reason is None
        assert res.top_k == 10
        assert res.count == 10
        assert len(res.recommendations) == 10
        assert res.inference_time_ms > 0

        # Check structure of items
        for rank_idx, item in enumerate(res.recommendations, start=1):
            assert item.rank == rank_idx
            assert item.song_id in music_svc.songs_by_id
            assert len(item.title) > 0
            assert len(item.artist_name) > 0

    def test_03_recommendations_exclude_seen_items(self, music_svc):
        """Verify recommended songs NEVER contain any song from user's training history."""
        user_id = "0023d50b3c99c4352585408b1b21f3762655f434"
        user_seen = music_svc.user_train_history[user_id]
        assert len(user_seen) > 0, "User must have seen items to test filtering."

        res = music_svc.get_recommendations(user_id=user_id, model_name="item_cf", top_k=20)
        rec_ids = {r.song_id for r in res.recommendations}

        leakage = rec_ids.intersection(user_seen)
        assert len(leakage) == 0, f"Data leakage detected! Recommended seen items: {leakage}"

    def test_04_cold_start_fallback_unknown_user(self, music_svc):
        """Verify an unknown user automatically triggers the Popularity Baseline fallback."""
        fake_user = "brand_new_cold_start_user_xyz_999"
        res = music_svc.get_recommendations(user_id=fake_user, model_name="item_cf", top_k=10)

        assert res.user_id == fake_user
        assert res.model_requested == "item_cf"
        assert res.model_used == "popularity"
        assert res.fallback_used is True
        assert res.fallback_reason is not None
        assert "no interaction history" in res.fallback_reason
        assert len(res.recommendations) == 10

    def test_05_input_validation_empty_user(self, music_svc):
        """Verify empty user_id raises ValueError."""
        with pytest.raises(ValueError) as exc:
            music_svc.get_recommendations(user_id="   ", model_name="item_cf", top_k=10)
        assert "user_id" in str(exc.value)

    def test_06_input_validation_invalid_top_k(self, music_svc):
        """Verify out-of-range top_k raises ValueError."""
        with pytest.raises(ValueError):
            music_svc.get_recommendations(user_id="0023d50b3c99c4352585408b1b21f3762655f434", top_k=0)

        with pytest.raises(ValueError):
            music_svc.get_recommendations(user_id="0023d50b3c99c4352585408b1b21f3762655f434", top_k=101)

    def test_07_input_validation_unknown_model(self, music_svc):
        """Verify invalid model name raises ModelNotFoundError."""
        with pytest.raises(ModelNotFoundError):
            music_svc.get_recommendations(
                user_id="0023d50b3c99c4352585408b1b21f3762655f434",
                model_name="fictional_model_slug",
            )

    def test_08_get_song_detail(self, music_svc):
        """Verify catalog song lookup and 404 behavior for unknown song."""
        valid_sid = "SOQMFQX12AB0189E9A"
        song = music_svc.get_song(valid_sid)
        assert song is not None
        assert song.song_id == valid_sid
        assert song.title == "Here I Am"
        assert song.artist_name == "SOJA"
        assert song.total_play_count is not None
        assert song.unique_listeners is not None

        invalid_song = music_svc.get_song("NON_EXISTENT_SONG_ID_999")
        assert invalid_song is None

    def test_09_get_user_history(self, music_svc):
        """Verify training history retrieval for active user and None for unknown user."""
        user_id = "0023d50b3c99c4352585408b1b21f3762655f434"
        history = music_svc.get_user_history(user_id)
        assert history is not None
        assert history.user_id == user_id
        assert history.total_songs == 4
        assert history.total_plays == 16
        assert len(history.history) == 4

        # Unknown user
        fake_history = music_svc.get_user_history("fake_user_123")
        assert fake_history is None

    def test_10_no_duplicate_recommendations(self, music_svc):
        """Verify recommendations list contains no duplicate song IDs."""
        user_id = "0023d50b3c99c4352585408b1b21f3762655f434"
        res = music_svc.get_recommendations(user_id=user_id, model_name="item_cf", top_k=50)
        song_ids = [r.song_id for r in res.recommendations]
        assert len(song_ids) == len(set(song_ids)), "Found duplicate song IDs in recommendations."
