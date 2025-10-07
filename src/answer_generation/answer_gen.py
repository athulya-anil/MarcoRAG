"""
Stage 8A — Answer Generation
Generates Groq LLaMA-3 answers from top-K retrieved chunks and saves them for evaluation.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from groq import Groq 


def load_retrieval(base_dir="retrieval_output"):
    """Load the most recent retrieval run."""
    run_dirs = sorted(Path(base_dir).glob("run_*"), key=os.path.getmtime)
    if not run_dirs:
        raise FileNotFoundError("No retrieval runs found.")
    latest = run_dirs[-1]
    with open(latest / "retrieval_results.json") as f:
        data = json.load(f)
    return data, latest


def generate_answer_groq(prompt, model="llama-3.3-70b-versatile"):
    """Query Groq LLaMA-3 with a factual prompt."""
    client = Groq() 
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a factual assistant. "
                    "Answer concisely and only using the provided context."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    return completion.choices[0].message.content.strip()


def run_answer_generation(top_k=3):
    retrieval_data, run_dir = load_retrieval()
    answer_dir = run_dir / "answers"
    answer_dir.mkdir(parents=True, exist_ok=True)
    out_path = answer_dir / "answers.json"

    all_answers = {}
    for qid, qinfo in retrieval_data.items():
        query = qinfo["query"]
        docs = [d["content"] for d in qinfo["retrieved_docs"][:top_k]]
        context = "\n\n".join(docs)
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer precisely."
        print(f"🧠 Generating answer for: {query[:60]}...")
        try:
            answer = generate_answer_groq(prompt)
        except Exception as e:
            print(f"⚠️ Groq generation failed: {e}")
            answer = "Generation error"
        all_answers[qid] = {"query": query, "answer": answer, "context": docs}

    with open(out_path, "w") as f:
        json.dump(all_answers, f, indent=2)

    print(f"✅ Answers saved to {out_path}")


if __name__ == "__main__":
    run_answer_generation()

