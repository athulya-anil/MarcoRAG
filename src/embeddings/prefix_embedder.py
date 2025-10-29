"""
Prefix-Fusion Embedder
----------------------
Injects metadata as formatted text prefixes before content.
"""

from src.embeddings.base_embedder import BaseEmbedder


class PrefixFusionEmbedder(BaseEmbedder):
    def prepare_text(self, chunk):
        # Handle metadata stored in 'metadata' dict or at top level
        metadata = chunk.get('metadata', chunk)

        prefix_parts = [
            f"[Category: {metadata.get('category', 'N/A')}]",
            f"[Keywords: {', '.join(metadata.get('keywords', []))}]",
            f"[Summary: {metadata.get('summary', '')}]"
        ]
        prefix = " ".join(prefix_parts)

        # Support both 'text' and 'content' field names
        content = chunk.get('text', chunk.get('content', ''))
        return f"{prefix} {content}"

