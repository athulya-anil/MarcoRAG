"""
Run Retrieval Pipeline
----------------------
Stage 4 of the RAG Engine:
Performs retrieval, computes grounding metrics (recall@k, precision@k, nDCG@k),
and evaluates LLM-generated answers for faithfulness and groundedness.

Usage:
    python -m src.retrieval.run_retrieval_pipeline
"""

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer
import faiss

# -------------------- Path Setup -------------------- #
ROOT_DIR = Path(__file__).resolve().parents[2]
METADATA_PATH = ROOT_DIR / "metadata_output" / "msmarco_passages_chunks_metadata.json"
EMBED_PATH = ROOT_DIR / "embeddings_output" / "prefix_msmarco_passages_chunks_metadata.json"

# -------------------- Imports -------------------- #
from src.retrieval.grounding_eval import recall_at_k, precision_at_k, ndcg_at_k
from src.retrieval.llm_answer_eval import evaluate_answer


def run_pipeline(metadata_path: str, embedding_path: str, query: str, top_k=5):
    """Run real retrieval and evaluation."""
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"❌ Metadata file not found: {metadata_path}")
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"❌ Embeddings file not found: {embedding_path}")

    # -------------------- Load Metadata & Embeddings -------------------- #
    with open(metadata_path, "r", encoding="utf-8") as f:
        docs_data = json.load(f)

    # Handle both list and dict formats for metadata
    if isinstance(docs_data, list):
        docs = {d["chunk_id"]: d for d in docs_data if "chunk_id" in d}
    elif isinstance(docs_data, dict):
        docs = docs_data
    else:
        raise ValueError("❌ Metadata file must contain a list or dict of chunks")

    with open(embedding_path, "r", encoding="utf-8") as f:
        embed_data = json.load(f)

    vectors = [e.get("vector") or e.get("embedding") for e in embed_data if "chunk_id" in e]
    chunk_ids = [e["chunk_id"] for e in embed_data if "chunk_id" in e]

    if not vectors or len(vectors[0]) == 0:
        raise ValueError("❌ No valid embeddings found in file.")

    embeddings = np.array(vectors, dtype="float32")
    print(f"✅ Loaded {len(embeddings)} embeddings (dim={embeddings.shape[1]})")

    # -------------------- Initialize FAISS -------------------- #
    index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    print(f"🔍 FAISS index built with {len(embeddings)} vectors.")

    # -------------------- Embed Query -------------------- #
    model = SentenceTransformer("all-MiniLM-L6-v2")  # 🔗 Match embedding stage
    query_vec = model.encode([query], normalize_embeddings=True)
    D, I = index.search(query_vec, top_k)

    retrieved = []
    for rank, idx in enumerate(I[0]):
        chunk_id = chunk_ids[idx]
        doc = docs.get(chunk_id, {})
        retrieved.append({
            "chunk_id": chunk_id,
            "text": doc.get("text") or doc.get("content", ""),
            "score": float(D[0][rank])
        })

    retrieved_ids = [r["chunk_id"] for r in retrieved]

    # -------------------- Load Latest Ground Truth -------------------- #
    gt_dir = ROOT_DIR / "retrieval_output"
    runs = sorted(gt_dir.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    latest_run = runs[0] if runs else None
    gt_path = latest_run / "ground_truth" / "gt.json" if latest_run else None

    if gt_path and gt_path.exists():
        with open(gt_path, "r", encoding="utf-8") as g:
            gt_data = json.load(g)

        # Try to find ground truth for this query - handle multiple query IDs
        ground_truth = []
        if isinstance(gt_data, dict):
            # Try common query IDs or use first available
            for query_id in ["query_1", next(iter(gt_data.keys()), None)]:
                if query_id and query_id in gt_data:
                    entry = gt_data[query_id]
                    ground_truth = entry.get("relevant_chunk_ids", [])
                    break

        print(f"🔎 Ground truth chunk IDs: {ground_truth[:5]}")
        print(f"🔎 Retrieved chunk IDs: {retrieved_ids[:5]}")

        print(f"✅ Loaded ground truth from {gt_path}")
    else:
        print("⚠️ No ground truth found, using fallback first-chunk ID")
        ground_truth = [next(iter(docs.keys()))] if docs else []

    # -------------------- Compute Metrics -------------------- #
    metrics = {
        "recall@k": recall_at_k(ground_truth, retrieved_ids, top_k),
        "precision@k": precision_at_k(ground_truth, retrieved_ids, top_k),
        "nDCG@k": ndcg_at_k(ground_truth, retrieved_ids, top_k),
    }

    sample_answer = "Retrieval-Augmented Generation uses retrieved context to ground LLM responses."
    answer_eval = evaluate_answer(sample_answer, retrieved)
    metrics.update(answer_eval)

    print(json.dumps(metrics, indent=2))
    return metrics, retrieved


# -------------------- Run Stage -------------------- #
if __name__ == "__main__":
    print("🚀 Running Stage 4: Real Retrieval and Evaluation Pipeline")
    print(f"📄 Using metadata file: {METADATA_PATH}\n")

    query = "What is Retrieval-Augmented Generation?"
    metrics, retrieved = run_pipeline(str(METADATA_PATH), str(EMBED_PATH), query=query)

    # -------------------- Save Results -------------------- #
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path("retrieval_output") / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    retrieval_results = {"query_1": {"query": query, "retrieved_docs": retrieved}}
    with open(run_dir / "retrieval_results.json", "w", encoding="utf-8") as f:
        json.dump(retrieval_results, f, indent=2)

    with open(run_dir / "metrics_overview.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Saved retrieval results and metrics to {run_dir}")
