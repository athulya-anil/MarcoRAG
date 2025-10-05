"""
TF-IDF Weighted Embedder
------------------------
Combines content and metadata fields with weighted importance.
"""

from src.embeddings.base_embedder import BaseEmbedder


class TFIDFEmbedder(BaseEmbedder):
    def __init__(self, model_name="all-MiniLM-L6-v2", content_weight=0.7, meta_weight=0.3):
        super().__init__(model_name)
        self.content_weight = content_weight
        self.meta_weight = meta_weight

    def prepare_text(self, chunk):
        content = chunk.get("content", "")
        metadata_text = " ".join([
            chunk.get("summary", ""),
            " ".join(chunk.get("keywords", [])),
            chunk.get("category", "")
        ])
        weighted_text = (
            f"{content * int(self.content_weight * 10)} "
            f"{metadata_text * int(self.meta_weight * 10)}"
        )
        return weighted_text

