"""Unit tests for Phase 4A: Neighborhood-based Collaborative Filtering.

Verifies:
1. UserBasedCF contract, prediction finite values, no NaNs.
2. ItemBasedCF contract, prediction finite values, no NaNs.
3. KNNBaseline contract, regularized biases, residual similarity, no NaNs.
4. Strict masking of train history across all three models.
5. Strict deduplication of recommended items.
6. Absence of test leakage during fitting.
7. Unobserved ratings are treated as missing, not as rating 0.
8. Unified candidate set policy enforced identically across models.
9. phase4a_metrics.csv integrity, row count, bounds [0, 1], 935 users.
10. Exact deterministic reproducibility.
"""

from pathlib import Path
import sys
import unittest

# Ensure REPO_ROOT is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from src.models.user_cf import UserBasedCF
from src.models.item_cf import ItemBasedCF
from src.models.knn_baseline import KNNBaseline
from src.evaluation.evaluator import TopNEvaluator

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"


class TestPhase4APipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
        cls.test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
        cls.movies_df = pd.read_csv(PROCESSED_DIR / "movies_clean.csv")

    def test_01_user_cf_contract_and_predict(self):
        """Criterion 1: UserBasedCF implements interface, yields finite non-NaN predictions."""
        model = UserBasedCF(k_neighbors=20).fit(self.train_df)
        self.assertTrue(model.is_fitted)

        # Predict for known user and movie
        score = model.predict(user_id=1, movie_id=50)
        self.assertIsInstance(score, float)
        self.assertFalse(np.isnan(score))
        self.assertFalse(np.isinf(score))

        # Recommend
        recs = model.recommend(user_id=1, top_n=10)
        self.assertEqual(len(recs), 10)
        self.assertEqual(len(set(recs)), 10)

    def test_02_item_cf_contract_and_predict(self):
        """Criterion 2: ItemBasedCF implements interface, yields finite non-NaN predictions."""
        model = ItemBasedCF(k_neighbors=20).fit(self.train_df)
        self.assertTrue(model.is_fitted)

        score = model.predict(user_id=1, movie_id=50)
        self.assertIsInstance(score, float)
        self.assertFalse(np.isnan(score))
        self.assertFalse(np.isinf(score))

        recs = model.recommend(user_id=1, top_n=10)
        self.assertEqual(len(recs), 10)
        self.assertEqual(len(set(recs)), 10)

    def test_03_knn_baseline_contract_and_predict(self):
        """Criterion 3: KNNBaseline implements regularized biases and residual similarity."""
        model = KNNBaseline(k_neighbors=20).fit(self.train_df)
        self.assertTrue(model.is_fitted)
        self.assertGreater(len(model.b_u), 0)
        self.assertGreater(len(model.b_i), 0)

        score = model.predict(user_id=1, movie_id=50)
        self.assertIsInstance(score, float)
        self.assertFalse(np.isnan(score))
        self.assertFalse(np.isinf(score))

        recs = model.recommend(user_id=1, top_n=10)
        self.assertEqual(len(recs), 10)
        self.assertEqual(len(set(recs)), 10)

    def test_04_no_seen_items_in_recommendations(self):
        """Criterion 4: Models never recommend an item already rated in Train history."""
        user_cf = UserBasedCF(k_neighbors=20).fit(self.train_df)
        item_cf = ItemBasedCF(k_neighbors=20).fit(self.train_df)
        knn_base = KNNBaseline(k_neighbors=20).fit(self.train_df)

        sample_users = [1, 25, 42, 100, 500]
        for uid in sample_users:
            train_seen = user_cf.user_train_history[uid]

            u_recs = user_cf.recommend(user_id=uid, top_n=10)
            self.assertEqual(len(set(u_recs).intersection(train_seen)), 0)

            i_recs = item_cf.recommend(user_id=uid, top_n=10)
            self.assertEqual(len(set(i_recs).intersection(train_seen)), 0)

            k_recs = knn_base.recommend(user_id=uid, top_n=10)
            self.assertEqual(len(set(k_recs).intersection(train_seen)), 0)

    def test_05_no_duplicate_items_in_recommendations(self):
        """Criterion 5: All recommendation lists contain strictly unique movie IDs."""
        models = [
            UserBasedCF(k_neighbors=20).fit(self.train_df),
            ItemBasedCF(k_neighbors=20).fit(self.train_df),
            KNNBaseline(k_neighbors=20).fit(self.train_df),
        ]
        for model in models:
            for uid in [1, 10, 943]:
                recs = model.recommend(user_id=uid, top_n=10)
                self.assertEqual(len(recs), len(set(recs)))

    def test_06_no_test_leakage_during_fit(self):
        """Criterion 6: Models are strictly fitted on Train without accessing Test interactions."""
        model = KNNBaseline().fit(self.train_df)
        for uid, train_seen in model.user_train_history.items():
            test_items = set(self.test_df[self.test_df["user_id"] == uid]["movie_id"])
            self.assertEqual(len(train_seen.intersection(test_items)), 0)

    def test_07_unobserved_ratings_not_treated_as_zero(self):
        """Criterion 7: Unobserved entries are not stored as explicit zero ratings."""
        user_cf = UserBasedCF().fit(self.train_df)
        # Verify sparse matrix NNZ equals exactly the number of observed training ratings
        self.assertEqual(user_cf.R_centered_T.nnz, len(self.train_df))

        item_cf = ItemBasedCF().fit(self.train_df)
        self.assertEqual(item_cf.R_centered.nnz, len(self.train_df))

    def test_08_candidate_set_consistency(self):
        """Criterion 8: Evaluator presents strictly identical candidate sets across models."""
        evaluator = TopNEvaluator(
            train_df=self.train_df,
            test_df=self.test_df,
            all_item_ids=self.movies_df["movie_id"].unique(),
            k=10,
        )
        self.assertEqual(len(evaluator.evaluation_users), 935)

        for uid in [1, 50, 935]:
            cands = evaluator.get_candidate_items(uid)
            train_seen = evaluator.user_train_history[uid]
            # Must equal all catalog items minus train seen
            self.assertEqual(set(cands), set(evaluator.all_items) - train_seen)

    def test_09_benchmark_csv_metrics_validity(self):
        """Criterion 9: phase4a_metrics.csv exists with 5 models, valid ranges, and split timings."""
        bench_csv = RESULTS_DIR / "phase4a_metrics.csv"
        self.assertTrue(bench_csv.is_file(), "phase4a_metrics.csv missing!")

        df = pd.read_csv(bench_csv)
        self.assertEqual(len(df), 5, "Expected 5 models in benchmark CSV")

        expected_models = {"Popularity", "Content-Based", "User-Based CF", "Item-Based CF", "KNNBaseline"}
        self.assertEqual(set(df["model"]), expected_models)

        # Check bounds [0, 1] for all metrics
        for col in ["precision_at_10", "recall_at_10", "f1_at_10", "ndcg_at_10", "hit_rate_at_10"]:
            self.assertTrue((df[col] >= 0.0).all())
            self.assertTrue((df[col] <= 1.0).all())

        # Check timings
        self.assertTrue((df["fit_time_seconds"] > 0.0).all())
        self.assertTrue((df["evaluate_time_seconds"] > 0.0).all())
        self.assertTrue((df["total_time_seconds"] > 0.0).all())
        self.assertTrue((df["evaluation_users"] == 935).all())

    def test_10_deterministic_reproducibility(self):
        """Criterion 10: Model recommendations are fully deterministic across runs."""
        model_1 = KNNBaseline(k_neighbors=20).fit(self.train_df)
        model_2 = KNNBaseline(k_neighbors=20).fit(self.train_df)

        recs_1 = model_1.recommend(user_id=1, top_n=10)
        recs_2 = model_2.recommend(user_id=1, top_n=10)
        self.assertEqual(recs_1, recs_2)


if __name__ == "__main__":
    unittest.main()
