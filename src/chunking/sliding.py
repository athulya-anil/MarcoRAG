# src/chunking/sliding.py
from .base import Chunker

class SlidingWindowChunker(Chunker):
    def __init__(self, window_size=800, overlap=100):
        self.window_size = window_size
        self.overlap = overlap

    def chunk(self, text: str, doc_name: str = "document"):
        chunks = []
        for i in range(0, len(text), self.window_size - self.overlap):
            sub = text[i : i + self.window_size]
            chunks.append({
                "chunk_id": f"{doc_name}_sliding_{len(chunks)}",
                "text": sub,
                "metadata": {
                    "num_words": len(sub.split()),
                    "strategy": "sliding",
                    "start_index": i,
                    "end_index": i + len(sub)
                }
            })
        return chunks

