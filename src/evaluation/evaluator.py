"""Unified Top-N Evaluation Harness for Recommender Systems.

This module enforces a strict, reproducible evaluation protocol:
- Standard Top-K evaluation (default K=10).
- Relevant ground truth: rating >= 4.0.
- Evaluated on users with at least 1 positive test item (935 users in ML-100K).
- Candidate items: all valid catalog items minus user's training history.
- User-level Macro-Averaging over all evaluated users.
"""

from pathlib import Path
import sys

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Any, Dict, List, Optional, Sequence, Set, Union
import logging
import time
import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    ndcg_at_k,
    hit_rate_at_k,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_RELEVANCE_THRESHOLD: float = 4.0
DEFAULT_K: int = 10


class TopNEvaluator:
    """Standardized Top-N Ranking Evaluator for all recommendation models."""

    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        all_item_ids: Optional[Sequence[Union[int, str]]] = None,
        relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
        k: int = DEFAULT_K,
    ) -> None:
        """Initialize evaluator with train and test data.

        Args:
            train_df: Training DataFrame containing ['user_id', 'movie_id', 'rating'].
            test_df: Testing DataFrame containing ['user_id', 'movie_id', 'rating'].
            all_item_ids: Optional full universe of item IDs. Defaults to union of train and test items.
            relevance_threshold: Minimum rating to consider an item relevant (default 4.0).
            k: Top-N cutoff rank (default 10).
        """
        self.relevance_threshold = float(relevance_threshold)
        self.k = int(k)

        # 1. Determine universal item catalog
        if all_item_ids is not None:
            self.all_items: Set[Union[int, str]] = set(all_item_ids)
        else:
            self.all_items = set(train_df["movie_id"]).union(set(test_df["movie_id"]))

        # 2. Build user train history (to strictly mask seen items)
        self.user_train_history: Dict[Union[int, str], Set[Union[int, str]]] = (
            train_df.groupby("user_id")["movie_id"].apply(set).to_dict()
        )

        # 3. Build user ground-truth positive test items (rating >= threshold)
        test_positives = test_df[test_df["rating"] >= self.relevance_threshold]
        self.user_test_positives: Dict[Union[int, str], Set[Union[int, str]]] = (
            test_positives.groupby("user_id")["movie_id"].apply(set).to_dict()
        )

        # 4. Target evaluation users: users in test with at least 1 positive item
        self.evaluation_users: List[Union[int, str]] = sorted(
            [uid for uid, items in self.user_test_positives.items() if len(items) >= 1]
        )

        logger.info(
            "TopNEvaluator initialized: %d catalog items, %d total test users, %d evaluation users (positives >= 1).",
            len(self.all_items),
            test_df["user_id"].nunique(),
            len(self.evaluation_users),
        )

    def get_candidate_items(self, user_id: Union[int, str]) -> List[Union[int, str]]:
        """Retrieve candidate items for a user (Catalog minus user's train history).

        Args:
            user_id: Target user ID.

        Returns:
            List[Union[int, str]]: Candidate item IDs.
        """
        seen_train = self.user_train_history.get(user_id, set())
        return [item for item in self.all_items if item not in seen_train]

    def evaluate_model(
        self,
        model: Any,
        model_name: Optional[str] = None,
        return_per_user: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate a recommendation model across all evaluation users.

        Args:
            model: An object with a `recommend(user_id, top_n, candidate_items)` method.
            model_name: Optional human-readable name of the model.
            return_per_user: Whether to include per-user metric breakdowns.

        Returns:
            Dict[str, Any]: Aggregated macro-average metrics, sample count, and execution time.
        """
        name = model_name or getattr(model, "name", model.__class__.__name__)
        logger.info("Starting Top-%d evaluation for model: %s...", self.k, name)

        start_time = time.time()

        precisions: List[float] = []
        recalls: List[float] = []
        f1s: List[float] = []
        ndcgs: List[float] = []
        hit_rates: List[float] = []
        per_user_records: List[Dict[str, Any]] = []

        for uid in self.evaluation_users:
            relevant = self.user_test_positives[uid]
            train_seen = self.user_train_history.get(uid, set())
            candidates = self.get_candidate_items(uid)

            # Query model for recommendations
            raw_recs = model.recommend(user_id=uid, top_n=self.k, candidate_items=candidates)

            # Sanitize recommendations:
            # 1. Deduplicate while preserving order
            cleaned_recs: List[Union[int, str]] = []
            seen_in_recs: Set[Union[int, str]] = set()
            for item in raw_recs:
                if item not in seen_in_recs and item not in train_seen and item in self.all_items:
                    seen_in_recs.add(item)
                    cleaned_recs.append(item)
                if len(cleaned_recs) == self.k:
                    break

            # Compute user-level metrics
            p = precision_at_k(cleaned_recs, relevant, k=self.k)
            r = recall_at_k(cleaned_recs, relevant, k=self.k)
            f1 = f1_at_k(cleaned_recs, relevant, k=self.k)
            ndcg = ndcg_at_k(cleaned_recs, relevant, k=self.k)
            hr = hit_rate_at_k(cleaned_recs, relevant, k=self.k)

            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)
            ndcgs.append(ndcg)
            hit_rates.append(hr)

            if return_per_user:
                per_user_records.append({
                    "user_id": uid,
                    "precision": p,
                    "recall": r,
                    "f1": f1,
                    "ndcg": ndcg,
                    "hit_rate": hr,
                    "num_relevant": len(relevant),
                    "num_recommended": len(cleaned_recs),
                })

        runtime = time.time() - start_time

        # Macro-Averaging over users
        summary = {
            "model": name,
            "precision_at_10": float(np.mean(precisions)),
            "recall_at_10": float(np.mean(recalls)),
            "f1_at_10": float(np.mean(f1s)),
            "ndcg_at_10": float(np.mean(ndcgs)),
            "hit_rate_at_10": float(np.mean(hit_rates)),
            "evaluation_users": len(self.evaluation_users),
            "runtime_seconds": round(runtime, 4),
        }

        if return_per_user:
            summary["per_user_df"] = pd.DataFrame(per_user_records)

        logger.info(
            "Evaluation complete for %s: P@10=%.4f, R@10=%.4f, F1@10=%.4f, NDCG@10=%.4f, HR@10=%.4f in %.2fs",
            name,
            summary["precision_at_10"],
            summary["recall_at_10"],
            summary["f1_at_10"],
            summary["ndcg_at_10"],
            summary["hit_rate_at_10"],
            runtime,
        )
        return summary
