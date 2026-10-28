"""Tests verifying MusicAI API contract alignment and backward-compatible aliases."""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_01_get_recommendations_contract():
    """Verify primary contract GET /api/music/recommendations."""
    res = client.get("/api/music/recommendations?user_id=0023d50b3c99c4352585408b1b21f3762655f434&model=item_cf&top_k=10")
    assert res.status_code == 200
    data = res.json()
    assert data["user_id"] == "0023d50b3c99c4352585408b1b21f3762655f434"
    assert data["model_requested"] == "item_cf"
    assert data["model_used"] == "item_cf"
    assert data["fallback_used"] is False
    assert data["count"] == 10
    assert len(data["recommendations"]) == 10
    rec0 = data["recommendations"][0]
    assert "rank" in rec0
    assert "song_id" in rec0
    assert "title" in rec0
    assert "artist_name" in rec0
    assert "score" in rec0


def test_02_get_recommendations_cold_start_fallback():
    """Verify cold-start user returns HTTP 200 with fallback_used=True and popularity model."""
    res = client.get("/api/music/recommendations?user_id=test-user-001&model=item_cf&top_k=10")
    assert res.status_code == 200
    data = res.json()
    assert data["user_id"] == "test-user-001"
    assert data["model_requested"] == "item_cf"
    assert data["model_used"] == "popularity"
    assert data["fallback_used"] is True
    assert "no interaction history" in data["fallback_reason"].lower()
    assert data["count"] == 10
    assert len(data["recommendations"]) == 10


def test_03_post_recommendations_alias():
    """Verify POST /api/music/recommendations accepts JSON body."""
    payload = {"user_id": "test-user-001", "model": "item_cf", "top_k": 5}
    res = client.post("/api/music/recommendations", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["count"] == 5
    assert data["fallback_used"] is True


def test_04_post_recommend_legacy_alias():
    """Verify POST /api/music/recommend legacy alias returns 200."""
    payload = {"user_id": "0023d50b3c99c4352585408b1b21f3762655f434", "model": "item_cf", "top_k": 5}
    res = client.post("/api/music/recommend", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["count"] == 5
    assert data["fallback_used"] is False


def test_05_get_user_history_standard_and_alias():
    """Verify both /users/{id}/history and /users/sample_users/{id}/history work."""
    uid = "0023d50b3c99c4352585408b1b21f3762655f434"
    res1 = client.get(f"/api/music/users/{uid}/history")
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["user_id"] == uid

    res2 = client.get(f"/api/music/users/sample_users/{uid}/history")
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["user_id"] == uid
    assert data1["total_songs"] == data2["total_songs"]
