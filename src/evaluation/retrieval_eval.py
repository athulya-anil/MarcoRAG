"""
Retrieval Evaluation (Stage 6)
------------------------------
Evaluates retrieval performance using latest run outputs.
Computes Precision@K, Recall@K, MRR, and NDCG@K metrics.
"""

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from src.retrieval.grounding_eval import recall_at_k, precision_at_k, ndcg_at_k


# -------------------- Helper -------------------- #
def load_latest_run(base_dir="retrieval_output"):
    """Automatically locate the latest retrieval run with a ground truth file."""
    run_dirs = sorted(Path(base_dir).glob("run_*"), key=os.path.getmtime, reverse=True)
    for run_dir in run_dirs:
        ret_path = run_dir / "retrieval_results.json"
        gt_path = run_dir / "ground_truth" / "gt.json"
        if ret_path.exists() and gt_path.exists():
            return ret_path, gt_path, run_dir
    raise FileNotFoundError("❌ No valid run found in retrieval_output/ with ground_truth/gt.json.")


# -------------------- Evaluation -------------------- #
def evaluate_retrieval():
    """Evaluate retrieval using ground truth and retrieved results."""
    ret_path, gt_path, run_dir = load_latest_run()

    print(f"📂 Evaluating run: {run_dir.name}")
    print(f"📄 Retrieval file: {ret_path}")
    print(f"📄 Ground truth file: {gt_path}\n")

    with open(ret_path, "r", encoding="utf-8") as f1:
        retrieval_data = json.load(f1)
    with open(gt_path, "r", encoding="utf-8") as f2:
        ground_truth_data = json.load(f2)

    results = {}
    k = 5

    for qid, qinfo in retrieval_data.items():
        retrieved = qinfo["retrieved_docs"]
        retrieved_ids = [r["chunk_id"] for r in retrieved]

        gt_entry = ground_truth_data.get(qid, {})
        ground_truth_ids = gt_entry.get("relevant_chunk_ids", [])

        if not ground_truth_ids:
            continue

        metrics = {
            "Precision@5": precision_at_k(ground_truth_ids, retrieved_ids, k),
            "Recall@5": recall_at_k(ground_truth_ids, retrieved_ids, k),
            "MRR": compute_mrr(ground_truth_ids, retrieved_ids),
            "NDCG@5": ndcg_at_k(ground_truth_ids, retrieved_ids, k)
        }
        results[qid] = metrics

    # Aggregate mean metrics
    if not results:
        print("⚠️  No valid results to evaluate")
        return {}

    avg_metrics = {metric: np.mean([res[metric] for res in results.values()])
                   for metric in results[next(iter(results))].keys()}

    # Save metrics
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    with open(eval_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(avg_metrics, f, indent=2)

    print("✅ Evaluation complete. Metrics saved to", eval_dir / "metrics.json")
    print(json.dumps(avg_metrics, indent=2))
    return avg_metrics


def compute_mrr(ground_truth_ids, retrieved_ids):
    """Compute Mean Reciprocal Rank."""
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in ground_truth_ids:
            return 1.0 / rank
    return 0.0


# -------------------- Run -------------------- #
if __name__ == "__main__":
    evaluate_retrieval()
