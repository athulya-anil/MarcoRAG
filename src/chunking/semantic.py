# src/chunking/semantic.py
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from .base import Chunker

class SemanticChunker(Chunker):
    def __init__(self, model_name="all-MiniLM-L6-v2",
                 percentile_threshold=85, min_chunk_size=3, max_chunk_size=15):
        self.model = SentenceTransformer(model_name)
        self.percentile_threshold = percentile_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

    def chunk(self, text: str, doc_name: str = "document"):
        sentences = sent_tokenize(text)
        if len(sentences) <= self.min_chunk_size:
            return [{
                "chunk_id": f"{doc_name}_semantic_0",
                "text": text,
                "metadata": {
                    "num_sentences": len(sentences),
                    "num_words": len(text.split()),
                    "strategy": "semantic",
                    "start_index": 0,
                    "end_index": len(sentences)
                }
            }]

        embeddings = self.model.encode(sentences)
        distances = [
            1 - np.dot(embeddings[i], embeddings[i + 1])
            / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1]))
            for i in range(len(embeddings) - 1)
        ]
        threshold = np.percentile(distances, self.percentile_threshold)
        breakpoints = [i for i, d in enumerate(distances) if d > threshold]

        chunks = []
        start = 0
        for bp in breakpoints:
            chunk_sents = sentences[start:bp+1]
            if len(chunk_sents) >= self.min_chunk_size:
                text_chunk = " ".join(chunk_sents)
                chunks.append({
                    "chunk_id": f"{doc_name}_semantic_{len(chunks)}",
                    "text": text_chunk,
                    "metadata": {
                        "num_sentences": len(chunk_sents),
                        "num_words": len(text_chunk.split()),
                        "strategy": "semantic",
                        "start_index": start,
                        "end_index": bp + 1
                    }
                })
            start = bp + 1
        if start < len(sentences):
            chunk_sents = sentences[start:]
            text_chunk = " ".join(chunk_sents)
            chunks.append({
                "chunk_id": f"{doc_name}_semantic_{len(chunks)}",
                "text": text_chunk,
                "metadata": {
                    "num_sentences": len(chunk_sents),
                    "num_words": len(text_chunk.split()),
                    "strategy": "semantic",
                    "start_index": start,
                    "end_index": len(sentences)
                }
            })
        return chunks

