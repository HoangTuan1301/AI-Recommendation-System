"""Integration Test Suite for CineAI FastAPI Backend."""

from pathlib import Path
import sys
import unittest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.main import app


class TestFastAPIBackend(unittest.TestCase):
    """Certifies FastAPI backend endpoints and model integration."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_health_endpoint(self):
        """Criterion 1: Health endpoint returns status healthy and catalog stats."""
        res = self.client.get("/api/v1/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["catalog_movies"], 1682)
        self.assertEqual(data["available_models"], 10)

    def test_02_models_registry(self):
        """Criterion 2: Models endpoint returns all 10 AI recommendation models."""
        res = self.client.get("/api/v1/models")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total"], 10)
        model_ids = [m["id"] for m in data["models"]]
        self.assertIn("neural_cf", model_ids)
        self.assertIn("hybrid", model_ids)
        self.assertIn("user_cf", model_ids)
        self.assertIn("popularity", model_ids)

    def test_03_movies_catalog_and_search(self):
        """Criterion 3: Movies catalog supports pagination, genre filtering, and search."""
        res = self.client.get("/api/v1/movies?page=1&limit=10")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total"], 1682)
        self.assertEqual(len(data["items"]), 10)

        # Search for Star Wars
        res_search = self.client.get("/api/v1/movies?q=Star Wars")
        self.assertEqual(res_search.status_code, 200)
        s_data = res_search.json()
        self.assertGreater(s_data["total"], 0)
        self.assertTrue(any("Star Wars" in it["title"] for it in s_data["items"]))

    def test_04_movie_detail(self):
        """Criterion 4: Movie detail endpoint returns metadata, rating distribution, and similar items."""
        res = self.client.get("/api/v1/movies/1")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["movie_id"], 1)
        self.assertTrue("Toy Story" in data["title"])
        self.assertIn("Animation", data["genres"])
        self.assertIn("rating_distribution", data)

    def test_05_users_list_and_profile(self):
        """Criterion 5: User profile endpoint returns stats strictly derived from train history."""
        res = self.client.get("/api/v1/users?page=1&limit=5")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total"], 943)

        res_prof = self.client.get("/api/v1/users/1/profile")
        self.assertEqual(res_prof.status_code, 200)
        p_data = res_prof.json()
        self.assertEqual(p_data["user_id"], 1)
        self.assertGreater(p_data["ratings_count"], 0)
        self.assertGreaterEqual(p_data["avg_rating"], 1.0)
        self.assertLessEqual(p_data["avg_rating"], 5.0)
        self.assertIsInstance(p_data["favorite_genres"], list)

    def test_06_metrics_leaderboard(self):
        """Criterion 6: Metrics endpoint returns valid benchmark metrics for all 10 models."""
        res = self.client.get("/api/v1/metrics")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total_models"], 10)
        self.assertIn("Neural CF", data["best_hit_rate_model"])

    def test_07_recommendations_inference_real_model(self):
        """Criterion 7: Recommendations endpoint generates Top 10 using checkpoint inference."""
        payload = {"user_id": 1, "model_id": "neural_cf", "top_n": 10}
        res = self.client.post("/api/v1/recommendations", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["user_id"], 1)
        self.assertEqual(data["top_n"], 10)
        self.assertEqual(len(data["recommendations"]), 10)

        # Check uniqueness
        rec_ids = [m["movie_id"] for m in data["recommendations"]]
        self.assertEqual(len(rec_ids), len(set(rec_ids)))

        # Check ranking order
        ranks = [m["rank"] for m in data["recommendations"]]
        self.assertEqual(ranks, list(range(1, 11)))

        # Check timing reported
        self.assertGreater(data["inference_time_ms"], 0)

    def test_08_no_seen_items_in_recommendations(self):
        """Criterion 8: Recommendations never contain items from user's train history."""
        # Get user 1's profile to extract top watched movies
        res_prof = self.client.get("/api/v1/users/1/profile")
        watched_sample = {m["movie_id"] for m in res_prof.json()["top_rated_movies"]}

        payload = {"user_id": 1, "model_id": "hybrid", "top_n": 20}
        res = self.client.post("/api/v1/recommendations", json=payload)
        rec_ids = {m["movie_id"] for m in res.json()["recommendations"]}
        self.assertEqual(len(watched_sample.intersection(rec_ids)), 0)

    def test_09_model_battle_compare(self):
        """Criterion 9: Compare endpoint computes Jaccard similarity and shared recommendations."""
        payload = {
            "user_id": 1,
            "model_a_id": "neural_cf",
            "model_b_id": "user_cf",
            "top_n": 10,
        }
        res = self.client.post("/api/v1/compare", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data["recommendations_a"]), 10)
        self.assertEqual(len(data["recommendations_b"]), 10)
        self.assertTrue(0.0 <= data["jaccard_similarity"] <= 1.0)
        self.assertEqual(data["overlap_count"], len(data["shared_movie_ids"]))


if __name__ == "__main__":
    unittest.main()
