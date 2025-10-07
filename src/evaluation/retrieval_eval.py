# src/evaluation/retrieval_eval.py
import os
import json
import math
from pathlib import Path
import numpy as np

def load_results(base_dir="retrieval_output"):
    run_dirs = sorted(Path(base_dir).glob("run_*"), key=os.path.getmtime)
    if not run_dirs:
        raise FileNotFoundError("No retrieval runs found.")
    latest = run_dirs[-1]
    ret_path = latest / "retrieval_results.json"
    gt_path = latest / "ground_truth" / "gt.json"
    with open(ret_path) as f1, open(gt_path) as f2:
        return json.load(f1), json.load(f2), latest

def precision_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    return len(set(retrieved_k) & set(relevant)) / k

def recall_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    return len(set(retrieved_k) & set(relevant)) / max(1, len(relevant))

def mrr(retrieved, relevant):
    for i, doc in enumerate(retrieved, 1):
        if doc in relevant:
            return 1 / i
    return 0.0

def ndcg_at_k(retrieved, relevant, k):
    dcg, idcg = 0.0, 0.0
    for i, doc in enumerate(retrieved[:k], 1):
        if doc in relevant:
            dcg += 1 / math.log2(i + 1)
    for i in range(min(k, len(relevant))):
        idcg += 1 / math.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_retrieval():
    retrieval, ground_truth, run_dir = load_results()
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_path = eval_dir / "metrics.json"

    metrics = {"Precision@5": [], "Recall@5": [], "MRR": [], "NDCG@5": []}

    for qid, gt in ground_truth.items():
        if qid not in retrieval:
            continue
        retrieved_docs = [d["content"] for d in retrieval[qid]["retrieved_docs"]]
        relevant_docs = [d["content"] for d in gt["docs"][:3]]  # top 3 as relevant
        metrics["Precision@5"].append(precision_at_k(retrieved_docs, relevant_docs, 5))
        metrics["Recall@5"].append(recall_at_k(retrieved_docs, relevant_docs, 5))
        metrics["MRR"].append(mrr(retrieved_docs, relevant_docs))
        metrics["NDCG@5"].append(ndcg_at_k(retrieved_docs, relevant_docs, 5))

    avg = {k: float(np.mean(v)) for k, v in metrics.items()}
    with open(out_path, "w") as f:
        json.dump({"per_query": metrics, "average": avg}, f, indent=2)
    print(f"✅ Evaluation complete. Metrics saved to {out_path}")
    print(json.dumps(avg, indent=2))

if __name__ == "__main__":
    evaluate_retrieval()

