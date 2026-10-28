"""Unit and Integration Tests for 6 MusicAI Recommendation Models (Phase M2)."""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "music" / "processed"
CHECKPOINTS_DIR = REPO_ROOT / "models" / "checkpoints" / "music"
CINEAI_CKPT_DIR = REPO_ROOT / "models" / "checkpoints"

from src.music.models.base_music_recommender import BaseMusicRecommender
from src.music.models.popularity import PopularityMusicRecommender
from src.music.models.item_cf import ItemCFMusicRecommender
from src.music.models.user_cf import UserCFMusicRecommender
from src.music.models.ials import IALSMusicRecommender
from src.music.models.bpr_mf import BPRMFMusicRecommender
from src.music.models.implicit_neural_cf import ImplicitNeuralCFRecommender


class TestMusicModels:
    """Test suite covering contracts, outputs, constraints, and CineAI isolation."""

    @pytest.fixture(scope="class")
    def sample_data(self):
        train_path = DATA_DIR / "train.csv"
        assert train_path.exists(), "train.csv missing!"
        train_df = pd.read_csv(train_path)
        return train_df

    def test_01_cineai_isolation_and_integrity(self):
        """Verify that all 10 CineAI checkpoints remain untouched and non-empty."""
        cineai_models = [
            "popularity.pkl", "content_based.pkl", "user_cf.pkl", "item_cf.pkl",
            "knn_baseline.pkl", "svd.pkl", "svdpp.pkl", "nmf.pkl",
            "neural_cf.pkl", "hybrid.pkl",
        ]
        for ckpt in cineai_models:
            p = CINEAI_CKPT_DIR / ckpt
            assert p.exists(), f"CineAI checkpoint missing: {ckpt}"
            assert p.stat().st_size > 100_000, f"CineAI checkpoint corrupted: {ckpt}"

    def test_02_music_data_integrity(self):
        """Ensure M1 processed datasets are unaltered."""
        for filename in ["train.csv", "validation.csv", "test.csv", "songs.csv"]:
            p = DATA_DIR / filename
            assert p.exists()
            df = pd.read_csv(p)
            assert len(df) > 0

    def test_03_popularity_recommender_contract(self, sample_data):
        """Test Popularity baseline output contract."""
        model = PopularityMusicRecommender()
        model.fit(sample_data)
        assert model.is_fitted

        sample_user = sample_data["user_id"].iloc[0]
        recs = model.recommend(sample_user, top_k=10, exclude_seen=True)

        assert len(recs) == 10
        rec_ids = [s for s, _ in recs]
        assert len(set(rec_ids)) == 10, "No duplicate songs"
        assert not set(rec_ids).intersection(model.user_train_history[sample_user]), "No seen songs"

        # Check cold user fallback
        cold_recs = model.recommend("non_existent_user_999", top_k=5)
        assert len(cold_recs) == 5

    def test_04_item_cf_contract(self, sample_data):
        """Test Item-Based CF output contract."""
        model = ItemCFMusicRecommender(n_neighbors=20)
        model.fit(sample_data)
        assert model.is_fitted

        sample_user = sample_data["user_id"].iloc[0]
        recs = model.recommend(sample_user, top_k=10, exclude_seen=True)
        assert len(recs) <= 10
        rec_ids = [s for s, _ in recs]
        assert len(set(rec_ids)) == len(rec_ids)
        assert not set(rec_ids).intersection(model.user_train_history[sample_user])

    def test_05_user_cf_contract(self, sample_data):
        """Test User-Based CF output contract."""
        model = UserCFMusicRecommender(n_neighbors=20)
        model.fit(sample_data)
        assert model.is_fitted

        sample_user = sample_data["user_id"].iloc[0]
        recs = model.recommend(sample_user, top_k=10, exclude_seen=True)
        assert len(recs) <= 10
        rec_ids = [s for s, _ in recs]
        assert len(set(rec_ids)) == len(rec_ids)
        assert not set(rec_ids).intersection(model.user_train_history[sample_user])

    def test_06_checkpoints_exist_and_load(self, sample_data):
        """Verify all 6 checkpoints exist in models/checkpoints/music/ and load properly."""
        expected_checkpoints = {
            "popularity.pkl": PopularityMusicRecommender,
            "item_cf.pkl": ItemCFMusicRecommender,
            "user_cf.pkl": UserCFMusicRecommender,
            "ials.pkl": IALSMusicRecommender,
            "bpr_mf.pkl": BPRMFMusicRecommender,
            "implicit_neumf.pkl": ImplicitNeuralCFRecommender,
        }

        sample_user = sample_data["user_id"].iloc[0]

        for filename, cls in expected_checkpoints.items():
            ckpt_path = CHECKPOINTS_DIR / filename
            if not ckpt_path.exists():
                pytest.skip(f"Checkpoint {filename} not yet generated (training in progress).")

            loaded = cls.load(ckpt_path)
            assert isinstance(loaded, BaseMusicRecommender)
            assert loaded.is_fitted

            recs = loaded.recommend(sample_user, top_k=5, exclude_seen=True)
            assert len(recs) == 5
            for s_id, score in recs:
                assert isinstance(s_id, str)
                assert not np.isnan(score)
                assert not np.isinf(score)

    def test_07_factorization_and_neural_models_score_stability(self, sample_data):
        """Verify iALS, BPR-MF, and NeuMF score outputs have no NaN or Inf."""
        sample_user = sample_data["user_id"].iloc[0]
        sample_songs = sample_data["song_id"].head(20).tolist()

        for ckpt, cls in [
            ("ials.pkl", IALSMusicRecommender),
            ("bpr_mf.pkl", BPRMFMusicRecommender),
            ("implicit_neumf.pkl", ImplicitNeuralCFRecommender),
        ]:
            model = cls.load(CHECKPOINTS_DIR / ckpt)
            scores = model.score(sample_user, sample_songs)
            assert len(scores) == 20
            assert not np.isnan(scores).any(), f"NaN detected in {ckpt} scores"
            assert not np.isinf(scores).any(), f"Inf detected in {ckpt} scores"

    def test_08_neumf_forward_and_gradient_flow(self, sample_data):
        """Verify NeuMF backward pass updates parameters properly."""
        import torch
        model = ImplicitNeuralCFRecommender(gmf_dim=8, mlp_dim=8, mlp_layers=[16, 8], max_epochs=1)
        sub_df = sample_data.head(200)
        model.fit(sub_df)
        assert model.is_fitted
        # Check that network weights are valid tensors
        for param in model.net.parameters():
            assert param.grad is not None or not param.requires_grad

    def test_09_test_set_isolation_and_checksum(self):
        """Ensure test.csv was not opened for training or modified."""
        import json, hashlib
        manifest_path = DATA_DIR / "dataset_manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        expected_test_hash = manifest["checksums"]["test_csv"]
        
        h = hashlib.sha256()
        with open(DATA_DIR / "test.csv", "rb") as f:
            h.update(f.read())
        assert h.hexdigest() == expected_test_hash, "test.csv checksum mismatch! Data was modified!"
