"""Integration Tests for MusicAI REST API and CineAI Regression."""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app


@pytest.fixture(scope="module")
def api_client():
    """TestClient fixture initialized with full multi-domain FastAPI app."""
    return TestClient(app)


class TestMusicAIEndpoints:
    """Test suite covering MusicAI API contracts, error handling, and multi-domain isolation."""

    def test_01_root_multi_domain(self, api_client):
        """Verify root endpoint exposes both CineAI and MusicAI domain paths."""
        res = api_client.get("/")
        assert res.status_code == 200
        data = res.json()
        assert "domains" in data
        assert "movies" in data["domains"]
        assert "music" in data["domains"]
        assert data["domains"]["music"]["service"] == "MusicAI"

    def test_02_music_health_endpoints(self, api_client):
        """Verify health check on both /api/music/health and /api/v1/music/health."""
        for path in ["/api/music/health", "/api/v1/music/health"]:
            res = api_client.get(path)
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "healthy"
            assert data["domain"] == "music"
            assert data["catalog_songs"] == 2706
            assert data["total_users"] == 2520
            assert data["available_models"] == 6
            assert data["default_model"] == "item_cf"

    def test_03_music_models_endpoints(self, api_client):
        """Verify model listing endpoint returns all 6 models with metadata."""
        for path in ["/api/music/models", "/api/v1/music/models"]:
            res = api_client.get(path)
            assert res.status_code == 200
            data = res.json()
            assert data["total"] == 6
            assert data["default_model"] == "item_cf"

            model_ids = [m["id"] for m in data["models"]]
            assert "item_cf" in model_ids
            assert "popularity" in model_ids
            assert "ials" in model_ids
            assert "user_cf" in model_ids
            assert "bpr_mf" in model_ids
            assert "implicit_neural_cf" in model_ids

    @pytest.mark.parametrize("model_slug", [
        "item_cf",
        "user_cf",
        "ials",
        "bpr_mf",
        "implicit_neural_cf",
        "popularity",
    ])
    def test_04_recommendations_all_six_models(self, api_client, model_slug):
        """Verify recommendation inference executes cleanly for all 6 models."""
        user_id = "0023d50b3c99c4352585408b1b21f3762655f434"
        res = api_client.get(f"/api/music/recommendations?user_id={user_id}&model={model_slug}&top_k=5")
        assert res.status_code == 200
        data = res.json()

        assert data["user_id"] == user_id
        assert data["model_requested"] == model_slug
        assert data["model_used"] == model_slug
        assert data["fallback_used"] is False
        assert data["count"] == 5
        assert len(data["recommendations"]) == 5
        assert data["inference_time_ms"] > 0

        for idx, rec in enumerate(data["recommendations"], start=1):
            assert rec["rank"] == idx
            assert len(rec["song_id"]) > 0
            assert len(rec["title"]) > 0
            assert len(rec["artist_name"]) > 0
            assert isinstance(rec["score"], float)

    def test_05_cold_start_fallback(self, api_client):
        """Verify unknown user returns Popularity fallback recommendations."""
        res = api_client.get("/api/music/recommendations?user_id=unknown_cold_user_999&top_k=5")
        assert res.status_code == 200
        data = res.json()

        assert data["fallback_used"] is True
        assert data["model_used"] == "popularity"
        assert "no interaction history" in data["fallback_reason"]
        assert len(data["recommendations"]) == 5

    def test_06_recommendations_validation_errors(self, api_client):
        """Verify HTTP 400/422 on invalid parameters."""
        user_id = "0023d50b3c99c4352585408b1b21f3762655f434"

        # Missing user_id -> 422 Unprocessable Entity
        res = api_client.get("/api/music/recommendations")
        assert res.status_code == 422

        # Invalid model name -> 400 Bad Request
        res = api_client.get(f"/api/music/recommendations?user_id={user_id}&model=non_existent_model")
        assert res.status_code == 400
        assert "Invalid model" in res.json()["detail"]

        # Invalid top_k (< 1) -> 422
        res = api_client.get(f"/api/music/recommendations?user_id={user_id}&top_k=0")
        assert res.status_code == 422

        # Invalid top_k (> 100) -> 422
        res = api_client.get(f"/api/music/recommendations?user_id={user_id}&top_k=101")
        assert res.status_code == 422

    def test_07_song_detail_endpoint(self, api_client):
        """Verify song catalog inspection and 404 on missing song."""
        valid_sid = "SOQMFQX12AB0189E9A"
        res = api_client.get(f"/api/music/songs/{valid_sid}")
        assert res.status_code == 200
        data = res.json()
        assert data["song_id"] == valid_sid
        assert data["title"] == "Here I Am"
        assert data["artist_name"] == "SOJA"
        assert data["total_play_count"] == 15
        assert data["unique_listeners"] == 7

        # Unknown song -> 404
        res_404 = api_client.get("/api/music/songs/UNKNOWN_FAKE_SONG_999")
        assert res_404.status_code == 404
        assert "not found" in res_404.json()["detail"]

    def test_08_user_history_endpoint(self, api_client):
        """Verify user training history retrieval and 404 on unknown user."""
        user_id = "0023d50b3c99c4352585408b1b21f3762655f434"
        res = api_client.get(f"/api/music/users/{user_id}/history")
        assert res.status_code == 200
        data = res.json()
        assert data["user_id"] == user_id
        assert data["total_songs"] == 4
        assert data["total_plays"] == 16
        assert len(data["history"]) == 4

        # Unknown user -> 404
        res_404 = api_client.get("/api/music/users/fake_cold_user_999/history")
        assert res_404.status_code == 404


class TestCineAIRegression:
    """Zero-regression test suite ensuring CineAI endpoints remain 100% functional."""

    def test_01_cineai_health(self, api_client):
        """Verify CineAI movie health endpoint."""
        res = api_client.get("/api/v1/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert data["service"] == "CineAI Recommendation Backend"
        assert data["catalog_movies"] == 1682
        assert data["available_models"] == 10

    def test_02_cineai_models(self, api_client):
        """Verify CineAI 10 movie recommendation models."""
        res = api_client.get("/api/v1/models")
        assert res.status_code == 200
        data = res.json()
        assert data["total"] == 10

    def test_03_cineai_movie_recommendations(self, api_client):
        """Verify CineAI movie recommendation POST endpoint."""
        payload = {"user_id": 1, "model_id": "item_cf", "top_n": 5}
        res = api_client.post("/api/v1/recommendations", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["user_id"] == 1
        assert data["model_id"] == "item_cf"
        assert len(data["recommendations"]) == 5

    def test_04_cineai_movie_detail(self, api_client):
        """Verify CineAI single movie retrieval."""
        res = api_client.get("/api/v1/movies/1")
        assert res.status_code == 200
        data = res.json()
        assert data["movie_id"] == 1
        assert "Toy Story" in data["title"]

    def test_05_cineai_metrics(self, api_client):
        """Verify CineAI offline benchmark metrics retrieval."""
        res = api_client.get("/api/v1/metrics")
        assert res.status_code == 200

    def test_06_cineai_compare(self, api_client):
        """Verify CineAI multi-model comparison POST endpoint."""
        payload = {"user_id": 1, "model_a_id": "popularity", "model_b_id": "item_cf", "top_n": 5}
        res = api_client.post("/api/v1/compare", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["user_id"] == 1
        assert "jaccard_similarity" in data
