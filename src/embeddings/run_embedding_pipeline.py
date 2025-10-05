"""
Stage 5: Embedding Generation Pipeline
--------------------------------------
Loads metadata-enriched chunks and generates embeddings.
"""
print(">>> RUNNING run_embedding_pipeline.py <<<")

import os
import json
from src.embeddings.embedder_factory import get_embedder


def run_pipeline(metadata_dir="metadata_output", output_dir="embeddings_output", embed_type="prefix"):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(metadata_dir):
        if not filename.endswith(".json"):
            continue

        with open(os.path.join(metadata_dir, filename), "r", encoding="utf-8") as f:
            chunks = json.load(f)

        embedder = get_embedder(embed_type)
        enriched = embedder.embed_chunks(chunks)

        out_path = os.path.join(output_dir, f"{embed_type}_{filename}")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(enriched, f, indent=2)

        print(f"✅ Saved embeddings to {out_path}")

    print(">>> Imports successful, starting pipeline <<<")


if __name__ == "__main__":
    print("🚀 Running Stage 5: Embedding Generation")
    run_pipeline(
        metadata_dir="metadata_output",
        output_dir="embeddings_output",
        embed_type="prefix"
    )
