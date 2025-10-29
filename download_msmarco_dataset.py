"""
Download and prepare MS MARCO dataset for RAG pipeline.
MS MARCO has passages (documents), queries, and human relevance judgments (ground truth).
"""

import json
import os
from pathlib import Path
from collections import defaultdict

def download_and_prepare_msmarco(output_dir="datasets", num_queries=100, num_passages=1000):
    """
    Download MS MARCO dataset - it has real passages and human relevance labels.

    Args:
        output_dir: Directory to save the prepared data
        num_queries: Number of queries to use
        num_passages: Number of passages (documents) to include
    """
    print("🔍 Downloading MS MARCO dataset...")

    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ Please install datasets: pip install datasets")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("📥 Loading MS MARCO passages and queries...")

    # Load MS MARCO v2.1
    # This has: queries, passages, and human relevance judgments
    try:
        dataset = load_dataset("ms_marco", "v2.1", split=f"train[:{num_queries}]")
    except:
        # Fallback to v1.1
        print("⚠️  Trying MS MARCO v1.1...")
        dataset = load_dataset("ms_marco", "v1.1", split=f"train[:{num_queries}]")

    print(f"📝 Processing {len(dataset)} samples...")

    # Collect unique passages and queries
    passages_dict = {}
    queries_list = []
    ground_truth = {}

    for idx, example in enumerate(dataset):
        query_id = f"query_{idx}"
        query_text = example.get('query', '')

        if not query_text:
            continue

        # Get passages for this query
        passages = example.get('passages', {})
        passage_texts = passages.get('passage_text', [])
        is_selected = passages.get('is_selected', [])

        # Find relevant passages (marked by human annotators)
        relevant_passage_ids = []

        for p_idx, (passage_text, selected) in enumerate(zip(passage_texts, is_selected)):
            passage_id = f"passage_{len(passages_dict)}"

            # Store passage if we haven't seen it
            if passage_text and passage_text not in [p['text'] for p in passages_dict.values()]:
                passages_dict[passage_id] = {
                    'id': passage_id,
                    'text': passage_text
                }

                # If this passage is relevant (selected by human), record it
                if selected == 1:
                    relevant_passage_ids.append(passage_id)

        # Only keep queries that have relevant passages
        if relevant_passage_ids:
            queries_list.append({
                'query_id': query_id,
                'query': query_text,
                'relevant_passages': relevant_passage_ids
            })

            # Ground truth: which passages are relevant for this query
            # This is HUMAN-ANNOTATED, not from retrieval!
            ground_truth[query_id] = {
                'query': query_text,
                'relevant_passage_ids': relevant_passage_ids
            }

        if len(passages_dict) >= num_passages:
            break

        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1} queries, collected {len(passages_dict)} passages...")

    # Save passages as text files in input_files/
    input_dir = Path("input_files")
    input_dir.mkdir(exist_ok=True)

    print(f"\n💾 Saving {len(passages_dict)} passages...")

    # Save all passages in one file for easier processing
    all_passages_path = input_dir / "msmarco_passages.txt"
    with open(all_passages_path, 'w', encoding='utf-8') as f:
        for passage_id, passage_data in passages_dict.items():
            f.write(f"[{passage_id}]\n")
            f.write(passage_data['text'])
            f.write("\n\n---\n\n")

    # Also save as JSON for reference
    passages_json_path = Path(output_dir) / "passages.json"
    with open(passages_json_path, 'w', encoding='utf-8') as f:
        json.dump(list(passages_dict.values()), f, indent=2)

    # Save queries
    queries_path = Path(output_dir) / "queries.json"
    with open(queries_path, 'w', encoding='utf-8') as f:
        json.dump(queries_list, f, indent=2)

    # Save ground truth (HUMAN ANNOTATIONS - this is the key!)
    gt_path = Path(output_dir) / "ground_truth.json"
    with open(gt_path, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\n✅ MS MARCO dataset prepared successfully!")
    print(f"   📁 Passages for pipeline: {all_passages_path}")
    print(f"   📋 Passages JSON: {passages_json_path}")
    print(f"   ❓ Queries: {queries_path}")
    print(f"   🎯 Ground truth (HUMAN-LABELED): {gt_path}")
    print(f"   📊 Total passages: {len(passages_dict)}")
    print(f"   ❓ Total queries: {len(queries_list)}")
    print(f"\n⚠️  IMPORTANT: The ground truth comes from HUMAN ANNOTATIONS")
    print(f"   not from your retrieval results, so evaluation will be meaningful!")

if __name__ == "__main__":
    download_and_prepare_msmarco(num_queries=100, num_passages=500)
