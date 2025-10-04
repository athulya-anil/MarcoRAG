# src/chunking/structural.py
from .base import Chunker
import re

class StructuralChunker(Chunker):
    def chunk(self, text: str, doc_name: str = "document"):
        sections = [s.strip() for s in re.split(r"\n{2,}|#+ ", text) if s.strip()]
        chunks = []
        for idx, section in enumerate(sections):
            chunks.append({
                "chunk_id": f"{doc_name}_structural_{idx}",
                "text": section,
                "metadata": {
                    "num_words": len(section.split()),
                    "strategy": "structural",
                    "start_index": idx,
                    "end_index": idx + 1
                }
            })
        return chunks

