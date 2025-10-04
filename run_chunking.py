# run_chunking.py - default - semantic
import os
import argparse
import json
import numpy as np
from src.chunking.factory import ChunkingFactory


def run_chunking(input_dir="input_files", output_dir="chunk_output", strategy="semantic"):
    """Run the chosen chunking strategy on all input text files."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"🧩 Using {strategy.upper()} chunking strategy\n")

    all_stats = []

    for fname in os.listdir(input_dir):
        if not fname.endswith(".txt"):
            continue

        path = os.path.join(input_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        chunker = ChunkingFactory.get_chunker(strategy)

        chunks = chunker.chunk(text, doc_name=os.path.splitext(fname)[0])
        out_path = os.path.join(output_dir, f"{os.path.splitext(fname)[0]}_{strategy}_chunks.json")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)

        avg_words = np.mean([c["metadata"]["num_words"] for c in chunks])
        avg_sentences = np.mean([c["metadata"].get("num_sentences", 0) for c in chunks])

        all_stats.append((fname, len(chunks), avg_words, avg_sentences))
        print(f"📄 {fname}: {len(chunks)} chunks → avg {avg_words:.1f} words, saved to {out_path}")

    print("\n📊 Chunking Summary")
    for fname, n_chunks, avg_words, avg_sent in all_stats:
        print(f"- {fname}: {n_chunks} chunks, avg {avg_words:.1f} words, avg {avg_sent:.1f} sentences")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run document chunking on text files.")
    parser.add_argument("--input_dir", type=str, default="input_files", help="Folder with .txt documents.")
    parser.add_argument("--output_dir", type=str, default="chunk_output", help="Where to save JSON chunks.")
    parser.add_argument(
        "--chunk_strategy",
        type=str,
        default="semantic",
        choices=["semantic", "structural", "sliding", "hybrid"],
        help="Chunking method to use (semantic = default, recommended)."
    )

    args = parser.parse_args()
    run_chunking(args.input_dir, args.output_dir, args.chunk_strategy)
