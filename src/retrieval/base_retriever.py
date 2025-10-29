"""
Base Retriever — common functionality for all retrieval strategies.
Handles safe embedding loading, FAISS index building, and vector search.
"""

import os
import json
import numpy as np
import faiss
from src.utils.validation import validate_embedding_structure, normalize_embedding_fields

# -----------------------------------------------------
# 🔹 Helper: Safe Embedding Loader
# -----------------------------------------------------
def load_embeddings(file_path: str):
    """Load embeddings safely with validation and return NumPy array + IDs + texts."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ids, texts, vectors = [], [], []
    invalid_count = 0

    for item in data:
        # Validate and normalize structure
        if not validate_embedding_structure(item):
            invalid_count += 1
            continue

        normalized = normalize_embedding_fields(item)

        ids.append(normalized.get("chunk_id", ""))
        texts.append(normalized.get("text", "") or normalized.get("content", ""))
        vectors.append(normalized["embedding"])

    if invalid_count > 0:
        print(f"⚠️  Skipped {invalid_count} invalid embedding entries")

    if not vectors:
        raise ValueError(f"❌ No valid embeddings found in {file_path}")

    embeddings = np.array(vectors, dtype="float32")
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    print(f"✅ Loaded {len(embeddings)} embeddings (dim={embeddings.shape[1]}) from {file_path}")
    return ids, texts, embeddings


# -----------------------------------------------------
# 🔹 Base Retriever Class
# -----------------------------------------------------
class BaseRetriever:
    def __init__(self, embedding_path: str, top_k: int = 5):
        self.embedding_path = embedding_path
        self.top_k = top_k
        self.ids, self.texts, self.embeddings = load_embeddings(embedding_path)
        self.index = None

    def build_index(self):
        """Build a FAISS index for similarity search using cosine similarity (inner product)."""
        dim = self.embeddings.shape[1]
        # Normalize embeddings for cosine similarity via inner product
        faiss.normalize_L2(self.embeddings)
        self.index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
        self.index.add(self.embeddings)
        print(f"🔍 Built FAISS index with {self.embeddings.shape[0]} vectors, dim={dim}")

    def retrieve(self, query_vector: np.ndarray):
        """Retrieve top-K most similar chunks given a query embedding."""
        if self.index is None:
            self.build_index()

        query_vector = np.array(query_vector, dtype="float32").reshape(1, -1)
        # Normalize query vector for cosine similarity
        faiss.normalize_L2(query_vector)
        D, I = self.index.search(query_vector, self.top_k)

        results = []
        for rank, idx in enumerate(I[0]):
            results.append({
                "rank": rank + 1,
                "chunk_id": self.ids[idx],
                "text": self.texts[idx],
                "distance": float(D[0][rank]),
            })
        return results

