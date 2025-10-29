"""
Grounding Evaluation
--------------------
Computes retrieval metrics: Recall@k, Precision@k, nDCG.
"""

import numpy as np

def recall_at_k(ground_truth, retrieved, k):
    """Calculate recall@k. Returns 0 if ground_truth is empty."""
    if not ground_truth:
        return 0.0
    return len(set(ground_truth) & set(retrieved[:k])) / len(ground_truth)

def precision_at_k(ground_truth, retrieved, k):
    """Calculate precision@k. Returns 0 if k is 0."""
    if k == 0:
        return 0.0
    return len(set(ground_truth) & set(retrieved[:k])) / k

def ndcg_at_k(ground_truth, retrieved, k):
    """Calculate NDCG@k. Returns 0 if ground_truth is empty."""
    if not ground_truth:
        return 0.0
    rel = [1 if r in ground_truth else 0 for r in retrieved[:k]]
    dcg = np.sum([r / np.log2(i + 2) for i, r in enumerate(rel)])
    idcg = np.sum([1 / np.log2(i + 2) for i in range(min(k, len(ground_truth)))])
    return dcg / idcg if idcg > 0 else 0.0

