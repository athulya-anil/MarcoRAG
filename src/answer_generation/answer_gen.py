"""
Stage 7: Answer Generation
--------------------------
Generates answers for retrieved queries using a selected LLM (Groq Llama-3, OpenAI, etc.)
and saves outputs for Stage 8 (answer quality evaluation).
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ------------------ Model Imports ------------------ #
try:
    from groq import Groq
    USE_GROQ = True
except ImportError:
    from openai import OpenAI
    USE_GROQ = False

# ------------------ Helpers ------------------ #
def load_latest_retrieval(base_dir="retrieval_output"):
    """Find the most recent retrieval run with results.json."""
    runs = sorted(Path(base_dir).glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for run in runs:
        ret_path = run / "retrieval_results.json"
        if ret_path.exists():
            return ret_path, run
    raise FileNotFoundError("❌ No retrieval_results.json found in retrieval_output/")

def build_prompt(query: str, docs: List[str]) -> str:
    """Format the retrieval context + question into a single prompt."""
    context = "\n\n".join([f"Context {i+1}:\n{doc}" for i, doc in enumerate(docs)])
    return (
        "You are a helpful and factual assistant.\n\n"
        f"{context}\n\n"
        f"Question: {query}\n\n"
        "Provide a concise and well-structured answer grounded in the above context."
    )

def generate_answer(prompt: str, temperature: float = 0.2, max_retries: int = 3) -> str:
    """Call Groq Llama-3 or OpenAI GPT to generate an answer with retry logic."""
    # Validate API key
    if USE_GROQ:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("❌ GROQ_API_KEY environment variable not set")
        client = Groq(api_key=api_key)
        model = "llama-3.1-8b-instant"
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("❌ OPENAI_API_KEY environment variable not set")
        client = OpenAI(api_key=api_key)
        model = "gpt-4o-mini"

    # Retry logic with exponential backoff
    for attempt in range(max_retries):
        try:
            if USE_GROQ:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=512,
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=512,
                )
            return response.choices[0].message.content.strip()

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"⚠️  API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"   Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"❌ API call failed after {max_retries} attempts: {e}")
                raise

# ------------------ Core Pipeline ------------------ #
def run_answer_generation():
    print("🚀 Running Stage 7: Answer Generation")

    # Locate latest retrieval run
    ret_path, run_dir = load_latest_retrieval()
    print(f"📄 Using retrieval results from: {ret_path}")

    with open(ret_path, "r", encoding="utf-8") as f:
        retrieval_data = json.load(f)

    answers_dir = run_dir / "answers"
    answers_dir.mkdir(parents=True, exist_ok=True)

    all_answers: Dict[str, Dict[str, str]] = {}
    for qid, qinfo in retrieval_data.items():
        query = qinfo.get("query", "")

        # Validate query and docs
        if not query or not query.strip():
            print(f"⚠️  Skipping {qid}: empty query")
            continue

        docs = [doc.get("content") or doc.get("text", "") for doc in qinfo.get("retrieved_docs", [])]
        docs = [d for d in docs if d.strip()]  # Filter out empty docs

        if not docs:
            print(f"⚠️  Skipping {qid}: no valid documents retrieved")
            continue

        top_docs = docs[:5]

        prompt = build_prompt(query, top_docs)
        print(f"🧠 Generating answer for: {query}")
        try:
            answer = generate_answer(prompt)
        except Exception as e:
            print(f"⚠️  LLM call failed for {qid}: {e}")
            answer = "Error generating answer."

        all_answers[qid] = {
            "query": query,
            "answer": answer,
            "used_docs": top_docs,
        }

    out_path = answers_dir / "answers.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_answers, f, indent=2)

    print(f"✅ Answers saved to {out_path}")
    return out_path

# ------------------ Run ------------------ #
if __name__ == "__main__":
    run_answer_generation()
