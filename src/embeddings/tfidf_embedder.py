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
        """
        Prepare weighted text by repeating content based on importance weights.
        Higher weights mean the text appears more times, increasing its TF-IDF score.
        """
        content = chunk.get("content", "")
        metadata_text = " ".join(filter(None, [
            chunk.get("summary", ""),
            " ".join(chunk.get("keywords", [])),
            chunk.get("category", "")
        ]))

        # Repeat text based on weights (multiply string by integer to repeat)
        content_repetitions = max(1, int(self.content_weight * 10))
        meta_repetitions = max(1, int(self.meta_weight * 10))

        # Join repeated texts with spaces
        weighted_parts = []
        if content:
            weighted_parts.extend([content] * content_repetitions)
        if metadata_text:
            weighted_parts.extend([metadata_text] * meta_repetitions)

        return " ".join(weighted_parts)

