"""
Prefix-Fusion Embedder
----------------------
Injects metadata as formatted text prefixes before content.
"""

from src.embeddings.base_embedder import BaseEmbedder


class PrefixFusionEmbedder(BaseEmbedder):
    def prepare_text(self, chunk):
        prefix_parts = [
            f"[Category: {chunk.get('category', 'N/A')}]",
            f"[Keywords: {', '.join(chunk.get('keywords', []))}]",
            f"[Summary: {chunk.get('summary', '')}]"
        ]
        prefix = " ".join(prefix_parts)
        return f"{prefix} {chunk.get('content', '')}"

