"""Standardized Top-N Evaluation Harness for MusicAI Models.

Enforces a reproducible, scientific Top-K evaluation protocol:
- Evaluates on Validation set (or Test set in M3).
- Positive ground truth: any interaction in eval_df (implicit binary preference p_ui=1).
- Candidate universe: all valid catalog songs (2,706) minus user's training history.
- Macro-Averaging over all eligible users with >= 1 positive ground truth song.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import logging
import time
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_ndcg_at_k(recommended_songs: List[str], ground_truth: Set[str], k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain at rank K."""
    hits = [1.0 if s in ground_truth else 0.0 for s in recommended_songs[:k]]
    if not hits or sum(hits) == 0.0:
        return 0.0

    dcg = sum(h / np.log2(idx + 2.0) for idx, h in enumerate(hits))
    # Ideal DCG: best possible ranking where min(len(ground_truth), k) hits appear at top
    n_ideal = min(len(ground_truth), k)
    idcg = sum(1.0 / np.log2(idx + 2.0) for idx in range(n_ideal))
    return float(dcg / idcg) if idcg > 0 else 0.0


def compute_mrr_at_k(recommended_songs: List[str], ground_truth: Set[str], k: int) -> float:
    """Calculate Mean Reciprocal Rank at rank K."""
    for idx, s in enumerate(recommended_songs[:k]):
        if s in ground_truth:
            return 1.0 / (idx + 1.0)
    return 0.0


class MusicTopNEvaluator:
    """Standardized Top-N Ranking Evaluator for MusicAI song recommenders."""

    def __init__(
        self,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        all_song_ids: Optional[Sequence[str]] = None,
        k: int = 10,
    ) -> None:
        self.k: int = int(k)
        if all_song_ids is not None:
            self.all_songs: List[str] = sorted(all_song_ids)
        else:
            self.all_songs = sorted(set(train_df["song_id"]).union(set(eval_df["song_id"])))

        # 1. Build training seen history
        self.user_train_history: Dict[str, Set[str]] = (
            train_df.groupby("user_id")["song_id"].apply(set).to_dict()
        )

        # 2. Build evaluation ground truth
        self.eval_ground_truth: Dict[str, Set[str]] = (
            eval_df.groupby("user_id")["song_id"].apply(set).to_dict()
        )

        # 3. Eligible users: users in eval_df with >= 1 positive song
        self.eligible_users: List[str] = sorted(
            [u for u, items in self.eval_ground_truth.items() if len(items) > 0]
        )
        logger.info(
            f"Initialized MusicTopNEvaluator: {len(self.eligible_users):,} eligible users, "
            f"{len(self.all_songs):,} total catalog songs, K={self.k}."
        )

    def evaluate(self, model: Any) -> Dict[str, float]:
        """Evaluate a model across all eligible evaluation users.

        Returns:
            Dict[str, float]: Macro-averaged metrics (HR@K, P@K, R@K, NDCG@K, MRR@K).
        """
        start_time = time.time()
        hr_list: List[float] = []
        prec_list: List[float] = []
        rec_list: List[float] = []
        ndcg_list: List[float] = []
        mrr_list: List[float] = []

        for user_id in self.eligible_users:
            truth = self.eval_ground_truth[user_id]
            seen = self.user_train_history.get(user_id, set())

            # Full-catalog candidate pool minus training seen items
            candidates = [s for s in self.all_songs if s not in seen]
            if not candidates:
                continue

            # Generate top-k recommendations
            recs = model.recommend(user_id=user_id, top_k=self.k, exclude_seen=True, candidate_songs=candidates)
            rec_song_ids = [s for s, _ in recs]

            # Metric calculations
            hit_count = sum(1 for s in rec_song_ids if s in truth)
            hr = 1.0 if hit_count > 0 else 0.0
            prec = hit_count / float(self.k)
            rec = hit_count / float(len(truth))
            ndcg = compute_ndcg_at_k(rec_song_ids, truth, self.k)
            mrr = compute_mrr_at_k(rec_song_ids, truth, self.k)

            hr_list.append(hr)
            prec_list.append(prec)
            rec_list.append(rec)
            ndcg_list.append(ndcg)
            mrr_list.append(mrr)

        elapsed = time.time() - start_time
        num_eval_users = len(hr_list)

        results = {
            f"Hit_Rate@{self.k}": float(np.mean(hr_list)) if num_eval_users > 0 else 0.0,
            f"Precision@{self.k}": float(np.mean(prec_list)) if num_eval_users > 0 else 0.0,
            f"Recall@{self.k}": float(np.mean(rec_list)) if num_eval_users > 0 else 0.0,
            f"NDCG@{self.k}": float(np.mean(ndcg_list)) if num_eval_users > 0 else 0.0,
            f"MRR@{self.k}": float(np.mean(mrr_list)) if num_eval_users > 0 else 0.0,
            "eval_users": num_eval_users,
            "eval_time_sec": round(elapsed, 2),
        }
        return results
