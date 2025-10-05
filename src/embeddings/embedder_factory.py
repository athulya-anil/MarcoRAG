"""
Embedder Factory
----------------
Returns embedder instance based on type.
"""

from src.embeddings.base_embedder import BaseEmbedder
from src.embeddings.tfidf_embedder import TFIDFEmbedder
from src.embeddings.prefix_embedder import PrefixFusionEmbedder


def get_embedder(embed_type: str = "naive") -> BaseEmbedder:
    if embed_type == "tfidf":
        return TFIDFEmbedder()
    elif embed_type == "prefix":
        return PrefixFusionEmbedder()
    else:
        # Default: basic content-only embedder
        class NaiveEmbedder(BaseEmbedder):
            def prepare_text(self, chunk):
                return chunk.get("content", "")
        return NaiveEmbedder()

