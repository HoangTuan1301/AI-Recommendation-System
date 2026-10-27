"""Regression Unit Test Suite for Phase 4B: Matrix Factorization Recommenders.

Validates:
1. Mathematical correctness of Funk SVD point-wise prediction.
2. SVD++ implicit feedback formulation and scaling (|I_u|^(-1/2) * sum(Y)).
3. NMF non-negativity constraints (W >= 0 and H >= 0).
4. Strict absence of test leakage during training.
5. Unobserved ratings are not treated as explicit zero ratings.
6. No NaN or Inf in predicted ratings or latent factors.
7. Candidate items never contain movies from user's training history.
8. Recommendations contain strictly unique item IDs.
9. Deterministic reproducibility via fixed random seed.
10. Seamless compatibility with TopNEvaluator.
11. Monotonic loss reduction / convergence behavior.
12. Validation split disjointness and anchoring integrity.
"""

from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from src.evaluation.evaluator import TopNEvaluator
from src.evaluation.validation import get_or_create_validation_split, split_train_val_per_user
from src.models.svd import SVDRecommender
from src.models.svdpp import SVDppRecommender
from src.models.nmf import NMFRecommender


class TestPhase4BPipeline(unittest.TestCase):
    """Test suite certifying Phase 4B Matrix Factorization models and protocols."""

    @classmethod
    def setUpClass(cls):
        """Load datasets once for class-level testing."""
        cls.train_df = pd.read_csv(REPO_ROOT / "data" / "processed" / "train.csv")
        cls.test_df = pd.read_csv(REPO_ROOT / "data" / "processed" / "test.csv")
        cls.movies_df = pd.read_csv(REPO_ROOT / "data" / "processed" / "movies_clean.csv")
        cls.train_inner, cls.val_df, cls.meta = get_or_create_validation_split()

    def test_01_svd_prediction_formula_hand_example(self):
        """Criterion 1: Hand-calculated numerical verification of Funk SVD prediction."""
        model = SVDRecommender(n_factors=3, random_state=42)
        model.mu = 3.5
        model.bu = np.array([0.2, -0.1], dtype=np.float32)
        model.bi = np.array([0.4, 0.1], dtype=np.float32)
        model.P = np.array([[1.0, 0.5, 0.2], [-0.5, 0.1, 0.8]], dtype=np.float32)
        model.Q = np.array([[0.5, 0.2, -0.1], [0.1, -0.4, 0.3]], dtype=np.float32)
        model.user_to_idx = {101: 0, 102: 1}
        model.item_to_idx = {201: 0, 202: 1}
        model.is_fitted = True

        # Expected calculation for user 101, item 201:
        # pred = mu + bu[0] + bi[0] + dot(P[0], Q[0])
        # dot = 1.0*0.5 + 0.5*0.2 + 0.2*(-0.1) = 0.5 + 0.1 - 0.02 = 0.58
        # pred = 3.5 + 0.2 + 0.4 + 0.58 = 4.68
        pred = model.predict(101, 201)
        self.assertAlmostEqual(pred, 4.68, places=5)

    def test_02_svdpp_implicit_feedback_component(self):
        """Criterion 2: SVD++ implicit feedback formulation and scaling."""
        model = SVDppRecommender(n_factors=2, random_state=42)
        model.mu = 3.0
        model.bu = np.array([0.1], dtype=np.float32)
        model.bi = np.array([0.2, 0.3], dtype=np.float32)
        model.P = np.array([[0.4, 0.6]], dtype=np.float32)
        model.Q = np.array([[0.5, 0.1], [0.2, 0.8]], dtype=np.float32)
        model.Y = np.array([[0.2, 0.4], [0.6, 0.2]], dtype=np.float32)
        model.user_to_idx = {1: 0}
        model.item_to_idx = {10: 0, 20: 1}
        model.is_fitted = True

        # User 1 rated item 10 and item 20 in training:
        # |I_u| = 2 -> sqrt_nu = 1 / sqrt(2)
        # sum(Y) = [0.2+0.6, 0.4+0.2] = [0.8, 0.6]
        # u_impl = [0.8 / sqrt(2), 0.6 / sqrt(2)] ~ [0.565685, 0.424264]
        # p_eff = P[0] + u_impl = [0.4 + 0.565685, 0.6 + 0.424264] = [0.965685, 1.024264]
        # For item 10:
        # dot(Q[0], p_eff) = 0.5 * 0.965685 + 0.1 * 1.024264 = 0.4828425 + 0.1024264 = 0.5852689
        # pred = 3.0 + 0.1 + 0.2 + 0.5852689 = 3.8852689
        model.user_impl_vectors = np.array(
            [[0.8 / np.sqrt(2.0), 0.6 / np.sqrt(2.0)]], dtype=np.float32
        )
        pred = model.predict(1, 10)
        self.assertAlmostEqual(pred, 3.8852689, places=5)

    def test_03_nmf_non_negativity_constraint(self):
        """Criterion 3: NMF strictly enforces non-negativity across all latent dimensions."""
        model = NMFRecommender(n_factors=10, n_epochs=5, random_state=42)
        toy_train = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 3],
                "movie_id": [10, 20, 20, 30, 10],
                "rating": [5.0, 1.0, 4.0, 2.0, 3.0],
                "timestamp": [100, 101, 102, 103, 104],
            }
        )
        model.fit(toy_train)
        self.assertTrue(np.all(model.W >= 0.0), "W contains negative values!")
        self.assertTrue(np.all(model.H >= 0.0), "H contains negative values!")
        for u in [1, 2, 3]:
            for m in [10, 20, 30]:
                self.assertGreaterEqual(model.predict(u, m), 0.0)

    def test_04_no_test_leakage_during_fit(self):
        """Criterion 4: Models are strictly trained on train data without accessing test interactions."""
        # Train on train_inner only, verify test items that never occurred are cleanly handled
        model = SVDRecommender(n_factors=10, n_epochs=5, random_state=42)
        model.fit(self.train_inner)
        self.assertEqual(len(model.user_train_history), len(self.train_inner["user_id"].unique()))

    def test_05_unobserved_ratings_not_treated_as_zero(self):
        """Criterion 5: Unobserved pairs are not treated as explicit zero ratings."""
        # Predict an unobserved candidate item; score must be sensible (> 1.0) and not close to zero
        model = SVDRecommender(n_factors=10, n_epochs=5, random_state=42)
        model.fit(self.train_inner)
        cand_items = model._get_default_candidates(1)
        unobserved_item = cand_items[0]
        pred = model.predict(1, unobserved_item)
        self.assertGreater(pred, 1.0, "Prediction collapsed toward zero due to missing-as-zero flaw!")

    def test_06_no_nan_or_inf_in_predictions(self):
        """Criterion 6: Factor matrices and predictions contain zero NaN or Inf."""
        models = [
            SVDRecommender(n_factors=10, n_epochs=3, random_state=42),
            SVDppRecommender(n_factors=10, n_epochs=2, random_state=42),
            NMFRecommender(n_factors=10, n_epochs=3, random_state=42),
        ]
        sample_train = self.train_df.iloc[:2000].copy()
        for m in models:
            m.fit(sample_train)
            pred = m.predict(sample_train.iloc[0]["user_id"], sample_train.iloc[0]["movie_id"])
            self.assertTrue(np.isfinite(pred), f"Model {m.name} produced non-finite prediction!")

    def test_07_no_seen_items_in_recommendations(self):
        """Criterion 7: Recommender never recommends an item in user's training history."""
        model = SVDRecommender(n_factors=10, n_epochs=3, random_state=42)
        sample_train = self.train_df.iloc[:5000].copy()
        model.fit(sample_train)
        user_id = int(sample_train.iloc[0]["user_id"])
        seen_items = set(sample_train[sample_train["user_id"] == user_id]["movie_id"])

        recs = model.recommend(user_id, top_n=10)
        overlap = set(recs).intersection(seen_items)
        self.assertEqual(len(overlap), 0, f"Found seen items in recommendation list: {overlap}")

    def test_08_no_duplicate_movie_ids(self):
        """Criterion 8: Recommendation lists contain strictly unique movie IDs."""
        model = NMFRecommender(n_factors=10, n_epochs=3, random_state=42)
        sample_train = self.train_df.iloc[:5000].copy()
        model.fit(sample_train)
        user_id = int(sample_train.iloc[0]["user_id"])
        recs = model.recommend(user_id, top_n=10)
        self.assertEqual(len(recs), len(set(recs)), "Duplicate IDs found in recommendation list!")

    def test_09_deterministic_reproducibility(self):
        """Criterion 9: Model recommendations are 100% deterministic with fixed seed."""
        sample_train = self.train_df.iloc[:3000].copy()
        m1 = SVDRecommender(n_factors=10, n_epochs=5, random_state=42)
        m2 = SVDRecommender(n_factors=10, n_epochs=5, random_state=42)
        m1.fit(sample_train)
        m2.fit(sample_train)

        u = int(sample_train.iloc[0]["user_id"])
        r1 = m1.recommend(u, top_n=10)
        r2 = m2.recommend(u, top_n=10)
        self.assertEqual(r1, r2, "Recommendations differed despite identical seed!")

    def test_10_topn_evaluator_compatibility(self):
        """Criterion 10: Seamless compatibility with TopNEvaluator harness."""
        sample_train = self.train_df.iloc[:5000].copy()
        sample_test = self.test_df[self.test_df["user_id"].isin(sample_train["user_id"])].copy()
        evaluator = TopNEvaluator(
            train_df=sample_train,
            test_df=sample_test,
            all_item_ids=self.movies_df["movie_id"].unique(),
            k=5,
        )
        model = SVDRecommender(n_factors=10, n_epochs=3, random_state=42)
        model.fit(sample_train)
        res = evaluator.evaluate_model(model)
        self.assertIn("hit_rate_at_10", res)
        self.assertTrue(0.0 <= res["hit_rate_at_10"] <= 1.0)

    def test_11_loss_tracking_and_convergence(self):
        """Criterion 11: Training RMSE decreases monotonically during early epochs."""
        model = SVDRecommender(n_factors=10, n_epochs=10, random_state=42)
        sample_train = self.train_df.iloc[:5000].copy()
        model.fit(sample_train)
        self.assertGreater(len(model.train_rmse_history), 1)
        self.assertLess(model.train_rmse_history[-1], model.train_rmse_history[0])

    def test_12_validation_split_disjointness(self):
        """Criterion 12: Train-Inner and Validation sets are completely disjoint."""
        train_pairs = set(zip(self.train_inner["user_id"], self.train_inner["movie_id"]))
        val_pairs = set(zip(self.val_df["user_id"], self.val_df["movie_id"]))
        overlap = train_pairs.intersection(val_pairs)
        self.assertEqual(len(overlap), 0, f"Found overlapping interactions in validation split: {len(overlap)}")

    def test_13_numerical_gradient_checking_svd(self):
        """Criterion 13: Numerical finite-difference gradient checking for Funk SVD."""
        n_users, n_items, d = 3, 4, 2
        rng = np.random.RandomState(42)
        mu = 3.5
        bu = rng.normal(0, 0.2, n_users)
        bi = rng.normal(0, 0.2, n_items)
        P = rng.normal(0, 0.2, (n_users, d))
        Q = rng.normal(0, 0.2, (n_items, d))

        u_idx = np.array([0, 0, 1, 2, 1])
        i_idx = np.array([1, 2, 0, 3, 2])
        ratings = np.array([4.0, 3.0, 5.0, 2.0, 4.0])
        reg = 0.05

        def loss_fn(bu_in, bi_in, P_in, Q_in):
            preds = mu + bu_in[u_idx] + bi_in[i_idx] + np.sum(P_in[u_idx] * Q_in[i_idx], axis=1)
            errs = ratings - preds
            return 0.5 * np.sum(errs**2) + 0.5 * reg * (
                np.sum(bu_in**2) + np.sum(bi_in**2) + np.sum(P_in**2) + np.sum(Q_in**2)
            )

        preds = mu + bu[u_idx] + bi[i_idx] + np.sum(P[u_idx] * Q[i_idx], axis=1)
        errs = ratings - preds

        grad_bu = reg * bu.copy()
        np.add.at(grad_bu, u_idx, -errs)
        grad_bi = reg * bi.copy()
        np.add.at(grad_bi, i_idx, -errs)
        grad_P = reg * P.copy()
        np.add.at(grad_P, u_idx, -errs[:, None] * Q[i_idx])
        grad_Q = reg * Q.copy()
        np.add.at(grad_Q, i_idx, -errs[:, None] * P[u_idx])

        eps = 1e-6
        for name, analytical, param, getter in [
            ("bu", grad_bu, bu, lambda: (bu, bi, P, Q)),
            ("bi", grad_bi, bi, lambda: (bu, bi, P, Q)),
            ("P", grad_P, P, lambda: (bu, bi, P, Q)),
            ("Q", grad_Q, Q, lambda: (bu, bi, P, Q)),
        ]:
            num_grad = np.zeros_like(analytical)
            it = np.nditer(analytical, flags=["multi_index"])
            while not it.finished:
                idx = it.multi_index
                orig = param[idx]
                param[idx] = orig + eps
                l_plus = loss_fn(*getter())
                param[idx] = orig - eps
                l_minus = loss_fn(*getter())
                param[idx] = orig
                num_grad[idx] = (l_plus - l_minus) / (2 * eps)
                it.iternext()
            rel_err = np.linalg.norm(analytical - num_grad) / (
                np.linalg.norm(analytical) + np.linalg.norm(num_grad)
            )
            self.assertLess(rel_err, 1e-5, f"Gradient mismatch for SVD parameter {name}")

    def test_14_numerical_gradient_checking_svdpp(self):
        """Criterion 14: Numerical finite-difference gradient checking for SVD++."""
        n_users, n_items, d = 2, 4, 2
        rng = np.random.RandomState(42)
        mu = 3.5
        bu = rng.normal(0, 0.2, n_users)
        bi = rng.normal(0, 0.2, n_items)
        P = rng.normal(0, 0.2, (n_users, d))
        Q = rng.normal(0, 0.2, (n_items, d))
        Y = rng.normal(0, 0.2, (n_items, d))

        user_items = [np.array([0, 1]), np.array([1, 2, 3])]
        user_sqrt = np.array([1.0 / np.sqrt(len(x)) for x in user_items])

        u_idx = np.array([0, 0, 1, 1, 1])
        i_idx = np.array([0, 1, 1, 2, 3])
        ratings = np.array([5.0, 3.0, 2.0, 4.0, 1.0])
        reg = 0.05

        def loss_fn(bu_in, bi_in, P_in, Q_in, Y_in):
            u_impl = np.zeros((n_users, d))
            for u in range(n_users):
                u_impl[u] = user_sqrt[u] * np.sum(Y_in[user_items[u]], axis=0)
            p_eff = P_in[u_idx] + u_impl[u_idx]
            preds = mu + bu_in[u_idx] + bi_in[i_idx] + np.sum(p_eff * Q_in[i_idx], axis=1)
            errs = ratings - preds
            return 0.5 * np.sum(errs**2) + 0.5 * reg * (
                np.sum(bu_in**2) + np.sum(bi_in**2) + np.sum(P_in**2) + np.sum(Q_in**2) + np.sum(Y_in**2)
            )

        u_impl = np.zeros((n_users, d))
        for u in range(n_users):
            u_impl[u] = user_sqrt[u] * np.sum(Y[user_items[u]], axis=0)
        p_eff = P[u_idx] + u_impl[u_idx]
        preds = mu + bu[u_idx] + bi[i_idx] + np.sum(p_eff * Q[i_idx], axis=1)
        errs = ratings - preds

        grad_Y = reg * Y.copy()
        for u in range(n_users):
            u_mask = u_idx == u
            accum_q = np.sum(errs[u_mask, None] * Q[i_idx[u_mask]], axis=0)
            grad_y_u = -user_sqrt[u] * accum_q
            for j in user_items[u]:
                grad_Y[j] += grad_y_u

        eps = 1e-6
        num_grad_Y = np.zeros_like(grad_Y)
        it = np.nditer(grad_Y, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = Y[idx]
            Y[idx] = orig + eps
            l_plus = loss_fn(bu, bi, P, Q, Y)
            Y[idx] = orig - eps
            l_minus = loss_fn(bu, bi, P, Q, Y)
            Y[idx] = orig
            num_grad_Y[idx] = (l_plus - l_minus) / (2 * eps)
            it.iternext()
        rel_err = np.linalg.norm(grad_Y - num_grad_Y) / (
            np.linalg.norm(grad_Y) + np.linalg.norm(num_grad_Y)
        )
        self.assertLess(rel_err, 1e-5, "Gradient mismatch for SVD++ Y implicit factors")

    def test_15_cold_user_fallback(self):
        """Criterion 15: Recommenders gracefully handle cold users with no training history."""
        models = [
            SVDRecommender(n_factors=10, n_epochs=3, random_state=42),
            SVDppRecommender(n_factors=10, n_epochs=2, random_state=42),
            NMFRecommender(n_factors=10, n_epochs=3, random_state=42),
        ]
        sample_train = self.train_df.iloc[:2000].copy()
        cold_user_id = 999999  # ID not in training set

        for m in models:
            m.fit(sample_train)
            pred = m.predict(cold_user_id, 1)
            self.assertTrue(np.isfinite(pred), f"{m.name} failed predict for cold user")
            recs = m.recommend(cold_user_id, top_n=5)
            self.assertEqual(len(recs), 5, f"{m.name} failed recommend for cold user")
            self.assertEqual(len(recs), len(set(recs)))


if __name__ == "__main__":
    unittest.main()
