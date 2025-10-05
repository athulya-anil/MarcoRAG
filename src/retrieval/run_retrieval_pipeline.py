"""
Run Retrieval Pipeline
----------------------
Stage 4 of the RAG Engine:
Performs retrieval, computes grounding metrics (recall@k, precision@k, nDCG@k),
and evaluates LLM-generated answers for faithfulness and groundedness.

Usage:
    python -m src.retrieval.run_retrieval_pipeline
    # or if inside src/retrieval/
    python run_retrieval_pipeline.py
"""

import os
import json
from pathlib import Path

# -------------------- Path Setup -------------------- #
# Automatically find project root regardless of where the script is run
ROOT_DIR = Path(__file__).resolve().parents[2]
METADATA_PATH = ROOT_DIR / "metadata_output" / "RAG_Google_Cloud_auto_metadata.json"

# -------------------- Imports -------------------- #
from src.retrieval.retriever_factory import get_retriever
from src.retrieval.grounding_eval import recall_at_k, precision_at_k, ndcg_at_k
from src.retrieval.llm_answer_eval import evaluate_answer


# -------------------- Core Pipeline -------------------- #
def run_pipeline(metadata_path: str, query: str, mode="keyword", top_k=5):
    """Run the retrieval and evaluation pipeline."""
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"❌ Metadata file not found: {metadata_path}")

    # Load enriched metadata
    with open(metadata_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    # Initialize retriever and perform retrieval
    retriever = get_retriever(mode, docs)
    retrieved = retriever.retrieve(query, top_k)
    retrieved_ids = [r["chunk_id"] for r in retrieved]

    # Dummy ground truth for initial testing
    ground_truth = [docs[0]["chunk_id"]]

    # Compute grounding metrics
    metrics = {
        "recall@k": recall_at_k(ground_truth, retrieved_ids, top_k),
        "precision@k": precision_at_k(ground_truth, retrieved_ids, top_k),
        "nDCG@k": ndcg_at_k(ground_truth, retrieved_ids, top_k),
    }

    # Evaluate LLM-generated answer
    sample_answer = "RAG combines retrieval and generation to improve LLM accuracy."
    answer_eval = evaluate_answer(sample_answer, retrieved)
    metrics.update(answer_eval)

    # Print nicely formatted JSON results
    print(json.dumps(metrics, indent=2))
    return metrics

if __name__ == "__main__":
    print("🚀 Running Stage 4: Retrieval and Evaluation Pipeline")
    print(f"📄 Using metadata file: {METADATA_PATH}\n")
    run_pipeline(str(METADATA_PATH), query="What is Retrieval-Augmented Generation?")
