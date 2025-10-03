# chunking.py
import os
import json
import argparse
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from sentence_transformers import SentenceTransformer

# Ensure NLTK resources
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


class SemanticChunker:
    def __init__(self, model_name="all-MiniLM-L6-v2", percentile_threshold=85, min_chunk_size=3, max_chunk_size=15):
        self.model = SentenceTransformer(model_name)
        self.percentile_threshold = percentile_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def chunk_document(self, text, doc_name):
        sentences = sent_tokenize(text)
        if len(sentences) <= self.min_chunk_size:
            return [{
                "chunk_id": f"{doc_name}_semantic_0",
                "text": text,
                "metadata": {"num_sentences": len(sentences), "num_words": len(text.split())}
            }]

        embeddings = self.model.encode(sentences)
        distances = [
            1 - np.dot(embeddings[i], embeddings[i+1]) /
            (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1]))
            for i in range(len(embeddings)-1)
        ]

        threshold = np.percentile(distances, self.percentile_threshold)
        breakpoints = [i for i, d in enumerate(distances) if d > threshold]

        chunks = []
        start = 0
        for bp in breakpoints:
            chunk_sents = sentences[start:bp+1]
            if len(chunk_sents) >= self.min_chunk_size:
                chunks.append({
                    "chunk_id": f"{doc_name}_semantic_{len(chunks)}",
                    "text": " ".join(chunk_sents),
                    "metadata": {"num_sentences": len(chunk_sents), "num_words": len(" ".join(chunk_sents).split())}
                })
            start = bp + 1

        if start < len(sentences):
            chunk_sents = sentences[start:]
            chunks.append({
                "chunk_id": f"{doc_name}_semantic_{len(chunks)}",
                "text": " ".join(chunk_sents),
                "metadata": {"num_sentences": len(chunk_sents), "num_words": len(" ".join(chunk_sents).split())}
            })

        return chunks


def run_chunking(input_dir="input_files", output_dir="chunk_output"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    chunker = SemanticChunker(
        model_name="all-MiniLM-L6-v2",
        percentile_threshold=95,   # fewer breakpoints = bigger chunks
        min_chunk_size=5,          # avoid tiny chunks
        max_chunk_size=40
    )
    all_stats = []

    for fname in os.listdir(input_dir):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(input_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunker.chunk_document(text, os.path.splitext(fname)[0])

        out_path = os.path.join(output_dir, f"{os.path.splitext(fname)[0]}_semantic_chunks.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)

        avg_size = sum(c["metadata"]["num_words"] for c in chunks) / len(chunks)
        all_stats.append((fname, len(chunks), avg_size))

        print(f"📄 {fname}: {len(chunks)} chunks (avg {avg_size:.1f} words) → saved to {out_path}")

    print("\n📊 Chunking Summary")
    for stat in all_stats:
        print(f"- {stat[0]}: {stat[1]} chunks, avg size {stat[2]:.1f} words")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="input_files")
    parser.add_argument("--output_dir", type=str, default="chunk_output")
    args = parser.parse_args()

    run_chunking(args.input_dir, args.output_dir)
