# src/metadata/metadata_eval.py
"""
Metadata Evaluation Module
--------------------------
Analyzes quality and completeness of LLM-generated metadata.

Usage:
    python run_metadata_eval.py --metadata_dir metadata_output --visualize
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

def analyze_metadata(file_path: str):
    """Compute statistics for a single metadata file."""
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
        "avg_completeness": np.mean(completeness),
        "unique_keywords": len(set(keywords)),
        "top_keywords": [w for w, _ in Counter(keywords).most_common(10)],
        "category_distribution": Counter(categories)
    }


def run_metadata_evaluation(metadata_dir: str = "metadata_output", visualize: bool = True):
    """Evaluate all metadata JSON files in a directory."""
    if not os.path.exists(metadata_dir):
        print(f"❌ Directory not found: {metadata_dir}")
        return

    files = [f for f in os.listdir(metadata_dir) if f.endswith(".json")]
    if not files:
        print("❌ No metadata JSON files found.")
        return

    results = []
    print(f"📊 Evaluating {len(files)} metadata files...\n")

    for fname in tqdm(files, ncols=80):
        stats = analyze_metadata(os.path.join(metadata_dir, fname))
        results.append(stats)

    df = pd.DataFrame(results)
    print("\n📈 Evaluation Summary:")
    print(df[["file", "num_chunks", "avg_completeness", "unique_keywords"]])

    # Save evaluation summary
    out_path = os.path.join(metadata_dir, "metadata_eval_summary.csv")
    df.to_csv(out_path, index=False)
    print(f"\n✅ Saved summary CSV → {out_path}")

    if visualize:
        visualize_results(df, metadata_dir)

    return df


def visualize_results(df: pd.DataFrame, out_dir: str):
    """Generate basic visualizations for metadata quality."""
    plt.figure(figsize=(10, 5))
    plt.bar(df["file"], df["avg_completeness"], color="teal")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Average Completeness")
    plt.title("Metadata Completeness by File")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "completeness_chart.png"))
    plt.close()

    print(f"📊 Saved visualization → {os.path.join(out_dir, 'completeness_chart.png')}")

