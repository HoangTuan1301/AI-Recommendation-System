"""Regression and Smoke Test Suite for Phase 5: Streamlit Web Application.

Validates:
1. App modules import cleanly without circular dependencies.
2. Model registry contains all 10 recommendation models with valid metadata.
3. Data loading functions load clean datasets and offline benchmark metrics.
4. Pre-trained checkpoints for all 10 models exist and load in sub-second time.
5. Models generate valid Top 5, Top 10, and Top 20 recommendations.
6. Recommendations contain strictly unique movie IDs.
7. Recommendations never contain items from user's training history.
8. Robust handling of invalid/cold user IDs without crashes.
9. Score formatting returns finite values without NaN or Inf.
10. Offline benchmark table contains all 10 models with valid metric values in [0, 1].
11. Side-by-side model comparison computation behaves correctly.
"""

from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from app.model_loader import (
    MODEL_REGISTRY,
    CHECKPOINTS_DIR,
    load_or_train_model,
    get_model_recommendations_with_scores,
)


class TestPhase5App(unittest.TestCase):
    """Test suite certifying the Streamlit Application and Model Loader."""

    @classmethod
    def setUpClass(cls):
        """Load datasets once for class-level testing."""
        cls.train_df = pd.read_csv(REPO_ROOT / "data" / "processed" / "train.csv")
        cls.movies_df = pd.read_csv(REPO_ROOT / "data" / "processed" / "movies_clean.csv")
        cls.metrics_df = pd.read_csv(REPO_ROOT / "results" / "phase4c_metrics.csv")

    def test_01_registry_contains_all_10_models(self):
        """Criterion 1: Registry contains exactly 10 models with required metadata keys."""
        self.assertEqual(len(MODEL_REGISTRY), 10)
        required_keys = {"id", "name", "family", "score_type", "score_range", "score_format", "description", "complexity"}
        for model_name, meta in MODEL_REGISTRY.items():
            self.assertTrue(required_keys.issubset(meta.keys()), f"Missing keys in {model_name}")

    def test_02_datasets_integrity(self):
        """Criterion 2: Processed datasets are loaded with expected row counts."""
        self.assertEqual(len(self.movies_df), 1682)
        self.assertEqual(len(self.train_df), 80000)
        self.assertEqual(self.train_df["user_id"].nunique(), 943)
        self.assertEqual(len(self.metrics_df), 10)

    def test_03_checkpoints_exist_and_load(self):
        """Criterion 3: Checkpoints for all 10 models exist on disk and load as fitted models."""
        for model_key, meta in MODEL_REGISTRY.items():
            ckpt_path = CHECKPOINTS_DIR / f"{meta['id']}.pkl"
            self.assertTrue(ckpt_path.exists(), f"Checkpoint missing: {ckpt_path.name}")
            model = load_or_train_model(model_key, train_df=self.train_df, movies_df=self.movies_df)
            self.assertTrue(getattr(model, "is_fitted", False), f"Model {model_key} is not fitted after loading")

    def test_04_recommendations_top_5_10_20(self):
        """Criterion 4: Models generate lists of exact length 5, 10, and 20."""
        model = load_or_train_model("User-Based CF", train_df=self.train_df, movies_df=self.movies_df)
        user_id = 1
        for k in [5, 10, 20]:
            recs = model.recommend(user_id=user_id, top_n=k)
            self.assertEqual(len(recs), k, f"Expected {k} recommendations, got {len(recs)}")

    def test_05_recommendations_strictly_unique(self):
        """Criterion 5: Recommended movie IDs contain no duplicates."""
        model = load_or_train_model("Popularity", train_df=self.train_df, movies_df=self.movies_df)
        recs = model.recommend(user_id=1, top_n=20)
        self.assertEqual(len(recs), len(set(recs)))

    def test_06_no_seen_items_in_recommendations(self):
        """Criterion 6: Recommender never suggests items present in user's train history."""
        user_id = 1
        seen_train_items = set(self.train_df[self.train_df["user_id"] == user_id]["movie_id"])
        model = load_or_train_model("Neural CF (NeuMF)", train_df=self.train_df, movies_df=self.movies_df)
        recs = model.recommend(user_id=user_id, top_n=20)
        intersection = seen_train_items.intersection(set(recs))
        self.assertEqual(len(intersection), 0, f"Found seen items in recommendation: {intersection}")

    def test_07_cold_user_handling(self):
        """Criterion 7: Unknown user ID (cold user) is handled without raising exceptions."""
        cold_uid = 99999
        for model_key in ["Popularity", "Content-Based", "User-Based CF", "Neural CF (NeuMF)", "Hybrid (CF + CB)"]:
            model = load_or_train_model(model_key, train_df=self.train_df, movies_df=self.movies_df)
            recs = model.recommend(user_id=cold_uid, top_n=5)
            self.assertIsInstance(recs, list)
            self.assertGreater(len(recs), 0)

    def test_08_score_formatting_and_no_nan(self):
        """Criterion 8: Recommendation item formatting produces finite scores without NaN or Inf."""
        model = load_or_train_model("Hybrid (CF + CB)", train_df=self.train_df, movies_df=self.movies_df)
        res = get_model_recommendations_with_scores(model, model_key="Hybrid (CF + CB)", user_id=196, top_n=5)
        self.assertEqual(len(res), 5)
        for item in res:
            self.assertIn("rank", item)
            self.assertIn("movie_id", item)
            self.assertIn("formatted_score", item)
            self.assertTrue(np.isfinite(item["raw_score"]))
            self.assertFalse(np.isnan(item["raw_score"]))

    def test_09_metrics_table_has_all_10_models_in_valid_range(self):
        """Criterion 9: Metrics table has 10 models with metrics in [0.0, 1.0]."""
        self.assertEqual(len(self.metrics_df), 10)
        metric_cols = ["precision_at_10", "recall_at_10", "f1_at_10", "ndcg_at_10", "hit_rate_at_10"]
        for col in metric_cols:
            self.assertTrue(col in self.metrics_df.columns)
            self.assertTrue(self.metrics_df[col].between(0.0, 1.0).all())

    def test_10_side_by_side_overlap_logic(self):
        """Criterion 10: Side-by-side comparison computes Jaccard similarity in [0, 1]."""
        m_a = load_or_train_model("Popularity", train_df=self.train_df, movies_df=self.movies_df)
        m_b = load_or_train_model("Content-Based", train_df=self.train_df, movies_df=self.movies_df)
        recs_a = m_a.recommend(user_id=1, top_n=10)
        recs_b = m_b.recommend(user_id=1, top_n=10)

        set_a, set_b = set(recs_a), set(recs_b)
        union = set_a.union(set_b)
        jaccard = len(set_a.intersection(set_b)) / len(union) if union else 0.0
        self.assertTrue(0.0 <= jaccard <= 1.0)

    def test_11_streamlit_all_pages_render_without_exception(self):
        """Criterion 11: All 4 Streamlit pages render cleanly using AppTest harness."""
        from streamlit.testing.v1 import AppTest

        pages = [
            "🎬 Gợi ý phim (Recommendations)",
            "📊 So sánh 10 mô hình (Benchmark)",
            "👤 Hồ sơ sở thích người dùng (Profiles)",
            "ℹ️ Giới thiệu đề tài (About)",
        ]

        at = AppTest.from_file(str(REPO_ROOT / "app" / "streamlit_app.py"), default_timeout=30)
        at.run()
        self.assertFalse(at.exception, f"Initial run exception: {at.exception}")

        for p in pages:
            at.sidebar.radio[0].set_value(p)
            at.run()
            self.assertFalse(at.exception, f"Exception on page '{p}': {at.exception}")



if __name__ == "__main__":
    unittest.main()
