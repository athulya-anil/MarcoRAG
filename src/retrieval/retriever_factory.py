"""
Retriever Factory
-----------------
Simplified version for MarcoRAG.
Supports FAISS-based retriever via BaseRetriever.
(You can later extend it to TF-IDF, Prefix Fusion, and Reranker retrievers.)
"""

import os
from typing import Dict, Any
from src.retrieval.base_retriever import BaseRetriever


def get_retriever(mode: str, docs=None, config: Dict[str, Any] = None):
    """
    Factory function to return a retriever instance.
    mode: "semantic", "faiss", "base" (default)
    """
    config = config or {}
    embedding_dir = config.get("embedding_dir", "embeddings_output")
    embedding_file = config.get(
        "embedding_file",
        "prefix_vertex_ai_corpus_semantic_metadata.json"
    )
    top_k = config.get("top_k", 5)

    embedding_path = os.path.join(embedding_dir, embedding_file)

    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"❌ Embedding file not found: {embedding_path}")

    if mode.lower() in ["semantic", "faiss", "base"]:
        print(f"✅ Using BaseRetriever (FAISS) for mode: {mode}")
        return BaseRetriever(embedding_path, top_k=top_k)

    else:
        raise ValueError(f"❌ Unknown retriever mode: {mode}")


def get_retrievers(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Multi-retriever version (placeholder for future hybrid support).
    Right now, returns only a FAISS/Base retriever.
    """
    retrievers = {}
    top_k = config.get("top_k", 5)
    embedding_dir = config.get("embedding_dir", "embeddings_output")

    # Default FAISS-based retriever
    retrievers["faiss"] = {
        "retriever": BaseRetriever(
            embedding_path=os.path.join(
                embedding_dir,
                config.get("embedding_file", "prefix_vertex_ai_corpus_semantic_metadata.json")
            ),
            top_k=top_k
        ),
        "name": "FAISS Retriever (Base)",
        "type": "faiss"
    }

    return retrievers

