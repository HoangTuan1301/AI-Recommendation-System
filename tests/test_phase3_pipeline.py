"""Comprehensive Unit Tests for Phase 3: Evaluation Framework & Baseline Recommenders.

Verifies:
1. Exact mathematical correctness of all 5 metrics using hand-calculated examples.
2. Edge cases for metrics (empty sets, zero division, k <= 0).
3. BaseRecommender contract enforcement.
4. PopularityRecommender fitting, scoring, and recommendation behavior.
5. ContentBasedRecommender fitting, TF-IDF vectorization, and profile construction.
6. Strict masking of train history (no seen items ever recommended).
7. Strict deduplication (recommendations contain no duplicate items).
8. Strict absence of test leakage during model fitting.
9. Exact reproducibility of recommendations and evaluation metrics.
10. Integrity and format of `results/phase3_baseline_metrics.csv`.
"""

from pathlib import Path
import math
import sys
import unittest

# Ensure REPO_ROOT is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    compute_all_topk_metrics,
)
from src.evaluation.evaluator import TopNEvaluator
from src.models.base_recommender import BaseRecommender
from src.models.popularity import PopularityRecommender
from src.models.content_based import ContentBasedRecommender

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"


class TestPhase3Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
        cls.test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
        cls.movies_df = pd.read_csv(PROCESSED_DIR / "movies_clean.csv")

    def test_01_metrics_hand_calculated_example(self):
        """Criterion 1: Exact numerical verification of metrics against analytical hand calculations."""
        # Setup: Recommended 5 items, 3 relevant items in ground truth
        recommended = [10, 20, 30, 40, 50]
        relevant = {20, 40, 99}  # items 20 and 40 hit at ranks 2 and 4. Item 99 not in recs.
        k = 5

        # 1. Precision@5 = 2 / 5 = 0.4000
        p = precision_at_k(recommended, relevant, k=k)
        self.assertAlmostEqual(p, 0.4, places=6)

        # 2. Recall@5 = 2 / 3 ≈ 0.666667
        r = recall_at_k(recommended, relevant, k=k)
        self.assertAlmostEqual(r, 2.0 / 3.0, places=6)

        # 3. F1@5 = 2 * (0.4 * 2/3) / (0.4 + 2/3) = 0.5000
        f1 = f1_at_k(recommended, relevant, k=k)
        self.assertAlmostEqual(f1, 0.5, places=6)

        # 4. Hit Rate@5 = 1.0
        hr = hit_rate_at_k(recommended, relevant, k=k)
        self.assertEqual(hr, 1.0)

        # 5. NDCG@5:
        # DCG@5 = 0 + 1/log2(3) + 0 + 1/log2(5) + 0
        dcg = (1.0 / math.log2(3)) + (1.0 / math.log2(5))
        # IDCG@5 = 1/log2(2) + 1/log2(3) + 1/log2(4)
        idcg = (1.0 / math.log2(2)) + (1.0 / math.log2(3)) + (1.0 / math.log2(4))
        expected_ndcg = dcg / idcg
        ndcg = ndcg_at_k(recommended, relevant, k=k)
        self.assertAlmostEqual(ndcg, expected_ndcg, places=6)

        # Verify batch helper
        metrics = compute_all_topk_metrics(recommended, relevant, k=k)
        self.assertAlmostEqual(metrics["precision_at_5"], 0.4, places=6)
        self.assertAlmostEqual(metrics["recall_at_5"], 2.0 / 3.0, places=6)
        self.assertAlmostEqual(metrics["f1_at_5"], 0.5, places=6)
        self.assertAlmostEqual(metrics["hit_rate_at_5"], 1.0, places=6)
        self.assertAlmostEqual(metrics["ndcg_at_5"], expected_ndcg, places=6)

    def test_02_metrics_edge_cases(self):
        """Criterion 2: Robust handling of empty sets, zero division, and extreme cutoffs."""
        # Empty relevant set
        self.assertEqual(precision_at_k([1, 2], set(), k=5), 0.0)
        self.assertEqual(recall_at_k([1, 2], set(), k=5), 0.0)
        self.assertEqual(f1_at_k([1, 2], set(), k=5), 0.0)
        self.assertEqual(hit_rate_at_k([1, 2], set(), k=5), 0.0)
        self.assertEqual(ndcg_at_k([1, 2], set(), k=5), 0.0)

        # Empty recommendations
        self.assertEqual(precision_at_k([], {1, 2}, k=5), 0.0)
        self.assertEqual(recall_at_k([], {1, 2}, k=5), 0.0)
        self.assertEqual(f1_at_k([], {1, 2}, k=5), 0.0)
        self.assertEqual(hit_rate_at_k([], {1, 2}, k=5), 0.0)
        self.assertEqual(ndcg_at_k([], {1, 2}, k=5), 0.0)

        # Non-positive K
        self.assertEqual(precision_at_k([1, 2], {1}, k=0), 0.0)
        self.assertEqual(recall_at_k([1, 2], {1}, k=-1), 0.0)

        # Perfect ranking
        perfect_recs = [1, 2, 3, 4]
        perfect_rel = {1, 2, 3, 4}
        self.assertEqual(ndcg_at_k(perfect_recs, perfect_rel, k=4), 1.0)
        self.assertEqual(precision_at_k(perfect_recs, perfect_rel, k=4), 1.0)
        self.assertEqual(recall_at_k(perfect_recs, perfect_rel, k=4), 1.0)

        # Disjoint (zero hits)
        self.assertEqual(hit_rate_at_k([10, 20], {1, 2}, k=2), 0.0)
        self.assertEqual(f1_at_k([10, 20], {1, 2}, k=2), 0.0)

    def test_03_base_recommender_contract(self):
        """Criterion 3: BaseRecommender abstract class cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            BaseRecommender()  # type: ignore

    def test_04_popularity_recommender_contract(self):
        """Criterion 4: PopularityRecommender implements contract, fits, and predicts."""
        model = PopularityRecommender(method="weighted_rating")
        model.fit(self.train_df)

        self.assertTrue(model.is_fitted)
        self.assertGreater(len(model.item_scores), 0)
        self.assertGreater(len(model.ranked_catalog), 0)

        # Recommend for user 1
        recs = model.recommend(user_id=1, top_n=10)
        self.assertEqual(len(recs), 10)
        self.assertEqual(len(set(recs)), 10, "Popularity recommendations must be unique")

        # Point prediction
        score = model.predict(user_id=1, movie_id=50)
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)

    def test_05_content_based_recommender_contract(self):
        """Criterion 5: ContentBasedRecommender implements contract, vectorizes, and predicts."""
        cb_model = ContentBasedRecommender(movies_df=self.movies_df)
        cb_model.fit(self.train_df)

        self.assertTrue(cb_model.is_fitted)
        self.assertIsNotNone(cb_model.item_features)
        self.assertIsNotNone(cb_model.user_profiles)

        recs = cb_model.recommend(user_id=1, top_n=10)
        self.assertEqual(len(recs), 10)
        self.assertEqual(len(set(recs)), 10, "Content-based recommendations must be unique")

        # Point prediction
        sim = cb_model.predict(user_id=1, movie_id=50)
        self.assertIsInstance(sim, float)
        self.assertTrue(-1.0 <= sim <= 1.0, f"Cosine similarity out of bounds: {sim}")

    def test_06_no_seen_items_in_recommendations(self):
        """Criterion 6: Recommender models never recommend an item present in user's train history."""
        pop = PopularityRecommender().fit(self.train_df)
        cb = ContentBasedRecommender(movies_df=self.movies_df).fit(self.train_df)

        sample_users = [1, 2, 5, 10, 50, 100]
        for uid in sample_users:
            seen_items = pop.user_train_history[uid]

            pop_recs = pop.recommend(user_id=uid, top_n=10)
            self.assertEqual(len(set(pop_recs).intersection(seen_items)), 0,
                             f"Popularity model recommended seen items for user {uid}")

            cb_recs = cb.recommend(user_id=uid, top_n=10)
            self.assertEqual(len(set(cb_recs).intersection(seen_items)), 0,
                             f"Content-based model recommended seen items for user {uid}")

    def test_07_no_duplicate_items_in_recommendations(self):
        """Criterion 7: Recommendation lists contain strictly unique movie IDs."""
        pop = PopularityRecommender().fit(self.train_df)
        cb = ContentBasedRecommender(movies_df=self.movies_df).fit(self.train_df)

        for uid in [1, 42, 943]:
            pop_recs = pop.recommend(user_id=uid, top_n=10)
            self.assertEqual(len(pop_recs), len(set(pop_recs)))

            cb_recs = cb.recommend(user_id=uid, top_n=10)
            self.assertEqual(len(cb_recs), len(set(cb_recs)))

    def test_08_no_test_leakage_during_fit(self):
        """Criterion 8: Models are fitted purely on train_df without accessing test_df."""
        pop = PopularityRecommender()
        pop.fit(self.train_df)
        # Verify user history in model contains only train items
        for uid, train_seen in pop.user_train_history.items():
            test_items = set(self.test_df[self.test_df["user_id"] == uid]["movie_id"])
            self.assertEqual(len(train_seen.intersection(test_items)), 0)

    def test_09_evaluator_and_benchmark_reproducibility(self):
        """Criterion 9: TopNEvaluator yields deterministic results matching benchmark CSV."""
        evaluator = TopNEvaluator(
            train_df=self.train_df,
            test_df=self.test_df,
            all_item_ids=self.movies_df["movie_id"].unique(),
            k=10,
        )

        self.assertEqual(len(evaluator.evaluation_users), 935, "Evaluation users count must be 935")

        pop = PopularityRecommender().fit(self.train_df)
        res = evaluator.evaluate_model(pop, model_name="Popularity_Baseline")

        # Load benchmark CSV
        bench_csv = RESULTS_DIR / "phase3_baseline_metrics.csv"
        self.assertTrue(bench_csv.is_file(), "phase3_baseline_metrics.csv does not exist!")

        bench_df = pd.read_csv(bench_csv)
        pop_row = bench_df[bench_df["model"] == "Popularity_Baseline"].iloc[0]

        self.assertAlmostEqual(res["precision_at_10"], pop_row["precision_at_10"], places=5)
        self.assertAlmostEqual(res["recall_at_10"], pop_row["recall_at_10"], places=5)
        self.assertAlmostEqual(res["f1_at_10"], pop_row["f1_at_10"], places=5)
        self.assertAlmostEqual(res["ndcg_at_10"], pop_row["ndcg_at_10"], places=5)
        self.assertAlmostEqual(res["hit_rate_at_10"], pop_row["hit_rate_at_10"], places=5)
        self.assertEqual(res["evaluation_users"], 935)

    def test_10_benchmark_csv_metrics_bounded(self):
        """Criterion 10: phase3_baseline_metrics.csv has valid columns, rows, and bounds [0, 1]."""
        bench_csv = RESULTS_DIR / "phase3_baseline_metrics.csv"
        self.assertTrue(bench_csv.is_file())

        df = pd.read_csv(bench_csv)
        self.assertEqual(len(df), 2, "Benchmark CSV should contain exactly 2 baseline models")
        expected_cols = [
            "model",
            "precision_at_10",
            "recall_at_10",
            "f1_at_10",
            "ndcg_at_10",
            "hit_rate_at_10",
            "evaluation_users",
            "runtime_seconds",
        ]
        self.assertEqual(list(df.columns), expected_cols)

        # Check bounds
        for metric in ["precision_at_10", "recall_at_10", "f1_at_10", "ndcg_at_10", "hit_rate_at_10"]:
            self.assertTrue((df[metric] >= 0.0).all())
            self.assertTrue((df[metric] <= 1.0).all())

        self.assertTrue((df["evaluation_users"] == 935).all())
        self.assertTrue((df["runtime_seconds"] > 0.0).all())


if __name__ == "__main__":
    unittest.main()
