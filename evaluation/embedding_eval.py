"""
Embedding Evaluation
--------------------
Computes similarity, clustering, and quality metrics.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def evaluate_embeddings(embeddings):
    mat = np.vstack([e["embedding"] for e in embeddings])
    sim = cosine_similarity(mat)
    avg_sim = np.mean(sim)
    print(f"🔍 Average intra-embedding similarity: {avg_sim:.4f}")
    return avg_sim

