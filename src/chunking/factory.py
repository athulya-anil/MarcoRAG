# src/chunking/factory.py
import numpy as np
from sentence_transformers import SentenceTransformer

from .semantic import SemanticChunker
from .structural import StructuralChunker
from .sliding import SlidingWindowChunker
from .hybrid import HybridChunker

from .semantic import SemanticChunker
from .structural import StructuralChunker
from .sliding import SlidingWindowChunker
from .hybrid import HybridChunker

class ChunkingFactory:
    @staticmethod
    def get_chunker(strategy: str):
        strategy = strategy.lower()
        if strategy == "semantic":
            return SemanticChunker()
        elif strategy == "structural":
            return StructuralChunker()
        elif strategy == "sliding":
            return SlidingWindowChunker()
        elif strategy == "hybrid":
            return HybridChunker()
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

    @staticmethod
    def _auto_select_chunker(text: str, **kwargs):
        """
        ML-based document classifier: distinguishes structured vs narrative text.
        Uses embeddings and basic linguistic features.
        """
        model = SentenceTransformer("all-MiniLM-L6-v2")

        # Simple feature extraction
        lines = text.split("\n")
        sentences = [s for s in text.split(".") if s.strip()]
        avg_line_length = np.mean([len(line) for line in lines if line.strip()]) if lines else 0
        avg_sentence_length = np.mean([len(s.split()) for s in sentences]) if sentences else 0

        # Embedding variance = coherence indicator
        sample_sents = sentences[:10] if len(sentences) > 10 else sentences
        if sample_sents:
            embeddings = model.encode(sample_sents)
            coherence_score = np.mean(np.var(embeddings, axis=0))
        else:
            coherence_score = 0.0

        # Lightweight “classification” heuristic
        # More variance → more narrative
        # Short lines, low variance → more structured
        if avg_line_length < 80 and coherence_score < 0.15:
            print("🔍 Auto-Selected: StructuralChunker (structured format detected)")
            return StructuralChunker(**kwargs)
        elif coherence_score > 0.30 or avg_sentence_length > 18:
            print("🔍 Auto-Selected: SemanticChunker (narrative text detected)")
            return SemanticChunker(**kwargs)
        else:
            print("🔍 Auto-Selected: HybridChunker (mixed content detected)")
            return HybridChunker(**kwargs)
