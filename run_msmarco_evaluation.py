"""
Run evaluation on MS MARCO dataset with REAL human-labeled ground truth.
This is the proper way to evaluate RAG - NOT circular validation!
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from src.retrieval.grounding_eval import recall_at_k, precision_at_k, ndcg_at_k

def load_ms_marco_data():
    """Load MS MARCO queries and ground truth."""
    queries_path = Path("datasets/queries.json")
    gt_path = Path("datasets/ground_truth.json")

    with open(queries_path, 'r') as f:
        queries = json.load(f)

    with open(gt_path, 'r') as f:
        ground_truth = json.load(f)

    return queries, ground_truth

def load_embeddings_and_metadata():
    """Load passage embeddings and metadata."""
    embed_path = Path("embeddings_output/prefix_msmarco_passages_chunks_metadata.json")
    metadata_path = Path("metadata_output/msmarco_passages_chunks_metadata.json")

    with open(embed_path, 'r') as f:
        embed_data = json.load(f)

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Extract vectors and IDs
    vectors = []
    chunk_ids = []
    for item in embed_data:
        if "embedding" in item:
            vectors.append(item["embedding"])
            chunk_ids.append(item["chunk_id"])

    embeddings = np.array(vectors, dtype="float32")

    # Create metadata lookup
    metadata_dict = {item["chunk_id"]: item for item in metadata}

    return embeddings, chunk_ids, metadata_dict

def retrieve_for_query(query, model, index, chunk_ids, top_k=5):
    """Retrieve top-k passages for a query."""
    query_vec = model.encode([query], normalize_embeddings=True)
    D, I = index.search(query_vec, top_k)

    retrieved_ids = [chunk_ids[idx] for idx in I[0]]
    scores = [float(D[0][i]) for i in range(len(I[0]))]

    return retrieved_ids, scores

def evaluate_retrieval():
    """Evaluate retrieval using MS MARCO ground truth."""
    print("🔍 Loading MS MARCO data...")
    queries, ground_truth = load_ms_marco_data()

    print("📊 Loading embeddings...")
    embeddings, chunk_ids, metadata_dict = load_embeddings_and_metadata()

    # Build FAISS index
    print("🔧 Building FAISS index...")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # Load model
    print("🤖 Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Evaluate each query
    print(f"\n📝 Evaluating {len(queries)} queries...")
    results = {}
    all_metrics = []

    top_k = 5

    for query_data in queries:
        query_id = query_data["query_id"]
        query_text = query_data["query"]

        # Skip if no ground truth for this query
        if query_id not in ground_truth:
            continue

        # Get ground truth relevant passages (HUMAN-LABELED!)
        gt_passage_ids = ground_truth[query_id]["relevant_passage_ids"]

        # Retrieve passages
        retrieved_ids, scores = retrieve_for_query(query_text, model, index, chunk_ids, top_k)

        # Compute metrics against REAL ground truth
        metrics = {
            "recall@5": recall_at_k(gt_passage_ids, retrieved_ids, top_k),
            "precision@5": precision_at_k(gt_passage_ids, retrieved_ids, top_k),
            "ndcg@5": ndcg_at_k(gt_passage_ids, retrieved_ids, top_k)
        }

        results[query_id] = {
            "query": query_text,
            "retrieved": retrieved_ids,
            "ground_truth": gt_passage_ids,
            "metrics": metrics
        }

        all_metrics.append(metrics)

        print(f"  {query_id}: R@5={metrics['recall@5']:.3f} P@5={metrics['precision@5']:.3f} NDCG@5={metrics['ndcg@5']:.3f}")

    # Compute average metrics
    avg_metrics = {
        "recall@5": np.mean([m["recall@5"] for m in all_metrics]),
        "precision@5": np.mean([m["precision@5"] for m in all_metrics]),
        "ndcg@5": np.mean([m["ndcg@5"] for m in all_metrics])
    }

    print(f"\n{'='*60}")
    print(f"📊 FINAL RESULTS (averaged over {len(all_metrics)} queries)")
    print(f"{'='*60}")
    print(f"  Recall@5:    {avg_metrics['recall@5']:.4f}")
    print(f"  Precision@5: {avg_metrics['precision@5']:.4f}")
    print(f"  NDCG@5:      {avg_metrics['ndcg@5']:.4f}")
    print(f"{'='*60}")
    print(f"\n✅ These scores are based on HUMAN GROUND TRUTH,")
    print(f"   not circular self-evaluation!")

    # Save results
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "msmarco_evaluation.json", 'w') as f:
        json.dump({
            "average_metrics": avg_metrics,
            "per_query_results": results,
            "num_queries": len(all_metrics)
        }, f, indent=2)

    print(f"\n💾 Results saved to {output_dir / 'msmarco_evaluation.json'}")

    return avg_metrics, results

if __name__ == "__main__":
    evaluate_retrieval()
