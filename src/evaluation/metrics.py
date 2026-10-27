"""Ranking and evaluation metrics for Top-N Recommender Systems.

This module provides standardized implementations of:
- Precision@K
- Recall@K
- F1@K
- NDCG@K (Normalized Discounted Cumulative Gain)
- Hit Rate@K
"""

from typing import Iterable, List, Sequence, Set, Union
import math
import numpy as np


def precision_at_k(
    recommended_items: Sequence[Union[int, str]],
    relevant_items: Union[Set[Union[int, str]], Sequence[Union[int, str]]],
    k: int = 10,
) -> float:
    """Compute Precision@K.

    Precision@K = |Recommended[:K] ∩ Relevant| / K

    Args:
        recommended_items: Ordered list of recommended item IDs.
        relevant_items: Set or sequence of ground-truth relevant item IDs.
        k: Cutoff rank (default 10).

    Returns:
        float: Precision@K in [0.0, 1.0].
    """
    if k <= 0:
        return 0.0

    relevant_set = set(relevant_items)
    if not relevant_set:
        return 0.0

    top_k = recommended_items[:k]
    hits = sum(1 for item in top_k if item in relevant_set)
    return float(hits / k)


def recall_at_k(
    recommended_items: Sequence[Union[int, str]],
    relevant_items: Union[Set[Union[int, str]], Sequence[Union[int, str]]],
    k: int = 10,
) -> float:
    """Compute Recall@K.

    Recall@K = |Recommended[:K] ∩ Relevant| / |Relevant|

    Args:
        recommended_items: Ordered list of recommended item IDs.
        relevant_items: Set or sequence of ground-truth relevant item IDs.
        k: Cutoff rank (default 10).

    Returns:
        float: Recall@K in [0.0, 1.0]. Returns 0.0 if relevant_items is empty.
    """
    if k <= 0:
        return 0.0

    relevant_set = set(relevant_items)
    if not relevant_set:
        return 0.0

    top_k = recommended_items[:k]
    hits = sum(1 for item in top_k if item in relevant_set)
    return float(hits / len(relevant_set))


def f1_at_k(
    recommended_items: Sequence[Union[int, str]],
    relevant_items: Union[Set[Union[int, str]], Sequence[Union[int, str]]],
    k: int = 10,
) -> float:
    """Compute F1@K (Harmonic mean of Precision@K and Recall@K).

    F1@K = 2 * (Precision@K * Recall@K) / (Precision@K + Recall@K)

    Args:
        recommended_items: Ordered list of recommended item IDs.
        relevant_items: Set or sequence of ground-truth relevant item IDs.
        k: Cutoff rank (default 10).

    Returns:
        float: F1@K in [0.0, 1.0].
    """
    p = precision_at_k(recommended_items, relevant_items, k=k)
    r = recall_at_k(recommended_items, relevant_items, k=k)

    if (p + r) == 0.0:
        return 0.0
    return float(2.0 * p * r / (p + r))


def hit_rate_at_k(
    recommended_items: Sequence[Union[int, str]],
    relevant_items: Union[Set[Union[int, str]], Sequence[Union[int, str]]],
    k: int = 10,
) -> float:
    """Compute Hit Rate@K (Binary indicator: 1.0 if at least one hit in Top-K else 0.0).

    HR@K = 1.0 if |Recommended[:K] ∩ Relevant| > 0 else 0.0

    Args:
        recommended_items: Ordered list of recommended item IDs.
        relevant_items: Set or sequence of ground-truth relevant item IDs.
        k: Cutoff rank (default 10).

    Returns:
        float: 1.0 if hit else 0.0.
    """
    if k <= 0:
        return 0.0

    relevant_set = set(relevant_items)
    if not relevant_set:
        return 0.0

    top_k = recommended_items[:k]
    for item in top_k:
        if item in relevant_set:
            return 1.0
    return 0.0


def ndcg_at_k(
    recommended_items: Sequence[Union[int, str]],
    relevant_items: Union[Set[Union[int, str]], Sequence[Union[int, str]]],
    k: int = 10,
) -> float:
    """Compute Normalized Discounted Cumulative Gain (NDCG@K) with binary relevance.

    DCG@K = sum_{i=1}^K rel_i / log2(i + 1)
    IDCG@K = sum_{i=1}^{min(K, |Relevant|)} 1 / log2(i + 1)
    NDCG@K = DCG@K / IDCG@K

    Args:
        recommended_items: Ordered list of recommended item IDs.
        relevant_items: Set or sequence of ground-truth relevant item IDs.
        k: Cutoff rank (default 10).

    Returns:
        float: NDCG@K in [0.0, 1.0].
    """
    if k <= 0:
        return 0.0

    relevant_set = set(relevant_items)
    if not relevant_set:
        return 0.0

    top_k = recommended_items[:k]

    # Calculate DCG@K
    dcg = 0.0
    for rank_idx, item in enumerate(top_k, start=1):
        if item in relevant_set:
            dcg += 1.0 / math.log2(rank_idx + 1)

    # Calculate IDCG@K (ideal ordering has all relevant items up to min(k, len(relevant_set)))
    num_ideal = min(k, len(relevant_set))
    idcg = sum(1.0 / math.log2(rank_idx + 1) for rank_idx in range(1, num_ideal + 1))

    if idcg == 0.0:
        return 0.0

    return float(dcg / idcg)


def compute_all_topk_metrics(
    recommended_items: Sequence[Union[int, str]],
    relevant_items: Union[Set[Union[int, str]], Sequence[Union[int, str]]],
    k: int = 10,
) -> dict:
    """Convenience helper to compute all 5 Top-K ranking metrics in one call.

    Args:
        recommended_items: Ordered list of recommended item IDs.
        relevant_items: Set or sequence of ground-truth relevant item IDs.
        k: Cutoff rank (default 10).

    Returns:
        dict: Mapping of metric names to values.
    """
    p = precision_at_k(recommended_items, relevant_items, k=k)
    r = recall_at_k(recommended_items, relevant_items, k=k)
    f1 = 2.0 * p * r / (p + r) if (p + r) > 0.0 else 0.0
    hr = hit_rate_at_k(recommended_items, relevant_items, k=k)
    ndcg = ndcg_at_k(recommended_items, relevant_items, k=k)

    return {
        f"precision_at_{k}": p,
        f"recall_at_{k}": r,
        f"f1_at_{k}": f1,
        f"ndcg_at_{k}": ndcg,
        f"hit_rate_at_{k}": hr,
    }
