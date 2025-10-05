"""
Base Embedder Interface
-----------------------
Defines the common structure for all embedding strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Dict
from sentence_transformers import SentenceTransformer


class BaseEmbedder(ABC):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    @abstractmethod
    def prepare_text(self, chunk: Dict) -> str:
        """Return text to embed given a metadata chunk."""
        pass

    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Generate embeddings for each chunk."""
        results = []
        for ch in chunks:
            text = self.prepare_text(ch)
            embedding = self.model.encode(text).tolist()
            ch["embedding"] = embedding
            results.append(ch)
        return results

