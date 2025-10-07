# src/retrieval/ground_truth_gen.py
import os
import json
import time
import torch
import numpy as np
from pathlib import Path
from sentence_transformers import CrossEncoder, util
from datetime import datetime

def load_retrieval_output(base_dir="retrieval_output"):
    run_dirs = sorted(Path(base_dir).glob("run_*"), key=os.path.getmtime)
    if not run_dirs:
        raise FileNotFoundError("No retrieval run found in retrieval_output/.")
    latest = run_dirs[-1]
    retrieval_file = latest / "retrieval_results.json"
    if not retrieval_file.exists():
        raise FileNotFoundError(f"No retrieval_results.json found in {latest}")
    with open(retrieval_file) as f:
        data = json.load(f)
    return data, latest

def rerank_cross_encoder(query, docs, model):
    pairs = [(query, d) for d in docs]
    scores = model.predict(pairs)
    return np.argsort(-np.array(scores)), scores.tolist()

def cosine_fallback(query_emb, doc_embs):
    sims = util.cos_sim(query_emb, doc_embs).cpu().numpy()[0]
    return np.argsort(-sims), sims.tolist()

def generate_ground_truth():
    print("🔍 Loading retrieval results...")
    retrieval_data, run_dir = load_retrieval_output()
    gt_dir = run_dir / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)
    gt_path = gt_dir / "gt.json"

    try:
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda" if torch.cuda.is_available() else "cpu")
        print("✅ Using cross-encoder reranker.")
        use_ce = True
    except Exception:
        from sentence_transformers import SentenceTransformer
        print("⚠️ Cross-encoder unavailable, falling back to cosine similarity.")
        use_ce = False
        embedder = SentenceTransformer("all-MiniLM-L6-v2")

    gt_data = {}
    for qid, qinfo in retrieval_data.items():
        query = qinfo["query"]
        docs = [d["content"] for d in qinfo["retrieved_docs"]]

        if not docs:
            continue

        if use_ce:
            order, scores = rerank_cross_encoder(query, docs, reranker)
        else:
            q_emb = embedder.encode(query, convert_to_tensor=True)
            d_embs = embedder.encode(docs, convert_to_tensor=True)
            order, scores = cosine_fallback(q_emb, d_embs)

        gt_data[qid] = {
            "query": query,
            "docs": [{"rank": int(i+1), "content": docs[idx], "score": float(scores[idx])}
                     for i, idx in enumerate(order)]
        }

    with open(gt_path, "w") as f:
        json.dump(gt_data, f, indent=2)
    print(f"✅ Ground truth saved to {gt_path}")

if __name__ == "__main__":
    generate_ground_truth()

