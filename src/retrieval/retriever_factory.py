"""
Retriever Factory Module
------------------------
Instantiates keyword, semantic, and hybrid retrievers for evaluating metadata-based RAG pipelines.
"""

from abc import ABC, abstractmethod
from typing import List, Dict

class BaseRetriever(ABC):
    """Abstract base class for retrievers."""
    def __init__(self, docs: List[Dict]):
        self.docs = docs

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5):
        pass


class KeywordRetriever(BaseRetriever):
    """Simple keyword-overlap retriever."""
    def retrieve(self, query: str, top_k: int = 5):
        q_terms = set(query.lower().split())
        scored = []
        for doc in self.docs:
            score = len(q_terms.intersection(set(doc["text"].lower().split())))
            scored.append((doc, score))
        ranked = sorted(scored, key=lambda x: x[1], reverse=True)
        return [d for d, _ in ranked[:top_k]]


def get_retriever(mode: str, docs: List[Dict]) -> BaseRetriever:
    if mode == "keyword":
        return KeywordRetriever(docs)
    else:
        raise ValueError(f"Unknown retriever mode: {mode}")

