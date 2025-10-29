"""
Download and prepare Natural Questions dataset for RAG pipeline.
This creates REAL ground truth from human annotations, not from retrieval results.
"""

import json
import os
from pathlib import Path

def download_and_prepare_nq(output_dir="datasets/natural_questions", num_samples=50):
    """
    Download Natural Questions dataset and prepare it for RAG.

    Args:
        output_dir: Directory to save the prepared data
        num_samples: Number of samples to download
    """
    print("🔍 Downloading Natural Questions dataset...")

    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ Please install datasets: pip install datasets")
        return

    # Download the simplified Natural Questions dataset
    print(f"📥 Loading {num_samples} samples from Natural Questions...")

    try:
        # Use the NQ-Open variant which is simpler and better for RAG
        dataset = load_dataset("nq_open", split="train")
    except Exception as e:
        print(f"⚠️  Could not load nq_open, trying alternative...")
        # Fallback to a different version
        dataset = load_dataset("google-research-datasets/natural_questions", split="train[:1000]", streaming=True)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # For NQ-Open format (question, answer pairs)
    documents = {}
    questions = []

    print(f"📝 Processing samples...")

    count = 0
    for idx, example in enumerate(dataset):
        if count >= num_samples:
            break

        # NQ-Open has: question, answer (list of possible answers)
        question_text = example.get('question', '')
        answers = example.get('answer', [])

        if not question_text or not answers:
            continue

        # Create a synthetic document ID (in real NQ, we'd have Wikipedia articles)
        # For NQ-Open, we'll create placeholder docs that we'll populate
        doc_id = f"nq_doc_{count}"

        # Store question with metadata
        questions.append({
            'query_id': f'query_{count}',
            'query': question_text,
            'document_id': doc_id,
            'answers': answers if isinstance(answers, list) else [answers],
            'has_answer': True
        })

        count += 1

        if count % 10 == 0:
            print(f"  Processed {count}/{num_samples} samples...")

    # Save queries and ground truth
    queries_path = Path(output_dir) / "queries.json"
    with open(queries_path, 'w', encoding='utf-8') as f:
        json.dump(questions, f, indent=2)

    # Create ground truth mapping (INDEPENDENT of retrieval)
    ground_truth = {}
    for q in questions:
        ground_truth[q['query_id']] = {
            'query': q['query'],
            'answers': q['answers'],
            # Note: We don't specify document IDs here because NQ-Open doesn't include them
            # The retrieval system will need to find relevant documents
        }

    gt_path = Path(output_dir) / "ground_truth.json"
    with open(gt_path, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\n✅ Dataset prepared successfully!")
    print(f"   📋 Queries: {queries_path}")
    print(f"   🎯 Ground truth: {gt_path}")
    print(f"   ❓ Total questions: {len(questions)}")

    return questions, ground_truth

if __name__ == "__main__":
    download_and_prepare_nq(num_samples=50)
