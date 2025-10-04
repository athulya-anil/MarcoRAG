# src/chunking/hybrid.py
from .base import Chunker
from .structural import StructuralChunker
from .semantic import SemanticChunker

class HybridChunker(Chunker):
    def __init__(self, model_name="all-MiniLM-L6-v2", similarity_threshold=0.7):
        self.structural = StructuralChunker()
        self.semantic = SemanticChunker(model_name=model_name)

    def chunk(self, text: str, doc_name: str = "document"):
        sections = self.structural.chunk(text, doc_name)
        all_chunks = []
        for section in sections:
            sub_chunks = self.semantic.chunk(section["text"], doc_name)
            all_chunks.extend(sub_chunks)
        # add metadata flag
        for c in all_chunks:
            c["metadata"]["strategy"] = "hybrid"
        return all_chunks
