# src/metadata/metadata_eval.py
"""
Lightweight Metadata Evaluation Module
--------------------------------------
Prints basic statistics about LLM-generated metadata.
No CSVs or plots are created.

Usage:
    python src/metadata/metadata_eval.py
"""

import os
import json
import numpy as np
from collections import Counter
from tqdm import tqdm

def analyze_metadata(file_path: str):
    """Compute quick summary statistics for a single metadata file."""
    with open(file_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    summaries, keywords, categories = [], [], []
    completeness = []

    for chunk in chunks:
        meta = chunk.get("metadata", {})
        filled = sum(bool(meta.get(k)) for k in ["summary", "entities", "keywords", "category"])
        completeness.append(filled / 4.0)
        summaries.append(meta.get("summary", ""))
        keywords.extend(meta.get("keywords", []))
        categories.append(meta.get("category", "Unknown"))

    return {
        "file": os.path.basename(file_path),
        "num_chunks": len(chunks),
        "avg_completeness": round(float(np.mean(completeness)) if completeness else 0, 3),
        "unique_keywords": len(set(keywords)),
        "top_keywords": [w for w, _ in Counter(keywords).most_common(5)],
        "top_categories": [c for c, _ in Counter(categories).most_common(3)]
    }

def run_metadata_evaluation(metadata_dir: str = "metadata_output"):
    """Evaluate all metadata JSON files in a directory and print summary."""
    if not os.path.exists(metadata_dir):
        print(f"❌ Directory not found: {metadata_dir}")
        return

    files = [f for f in os.listdir(metadata_dir) if f.endswith(".json")]
    if not files:
        print("❌ No metadata JSON files found.")
        return

    print(f"📊 Evaluating {len(files)} metadata files...\n")

    for fname in tqdm(files, ncols=80):
        file_path = os.path.join(metadata_dir, fname)
        stats = analyze_metadata(file_path)

        print(f"\n📘 {stats['file']}")
        print(f"   • Chunks: {stats['num_chunks']}")
        print(f"   • Avg completeness: {stats['avg_completeness']}")
        print(f"   • Unique keywords: {stats['unique_keywords']}")
        print(f"   • Top keywords: {', '.join(stats['top_keywords']) or '—'}")
        print(f"   • Top categories: {', '.join(stats['top_categories']) or '—'}")

    print("\n✅ Metadata evaluation completed.\n")

if __name__ == "__main__":
    run_metadata_evaluation("metadata_output")
