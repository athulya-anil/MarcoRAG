"""
Stage 8B — Answer Quality Evaluation
Evaluates each LLM answer for faithfulness, completeness, and hallucination rate.
"""

import os
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer, util


def load_answers(base_dir="retrieval_output"):
    run_dirs = sorted(Path(base_dir).glob("run_*"), key=os.path.getmtime)
    if not run_dirs:
        raise FileNotFoundError("No answer runs found.")
    latest = run_dirs[-1]
    with open(latest / "answers" / "answers.json") as f:
        answers = json.load(f)
    return answers, latest


def cosine_similarity(a, b, model):
    e1, e2 = model.encode(a, convert_to_tensor=True), model.encode(b, convert_to_tensor=True)
    return float(util.cos_sim(e1, e2))


def evaluate_answers():
    answers, run_dir = load_answers()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    out_dir = run_dir / "answers"
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / "answer_eval.json"

    results = {}
    for qid, entry in answers.items():
        query = entry["query"]
        answer = entry["answer"]
        context = " ".join(entry["context"])

        faithfulness = cosine_similarity(answer, context, model)
        completeness = cosine_similarity(query + " " + context, answer, model)
        hallucination = max(0.0, 1.0 - faithfulness)

        results[qid] = {
            "query": query,
            "faithfulness": round(faithfulness, 3),
            "completeness": round(completeness, 3),
            "hallucination": round(hallucination, 3),
        }

    avg = {m: float(np.mean([v[m] for v in results.values()])) for m in ["faithfulness", "completeness", "hallucination"]}

    with open(out_path, "w") as f:
        json.dump({"per_query": results, "average": avg}, f, indent=2)

    print(f"✅ Answer evaluation saved to {out_path}")
    print(json.dumps(avg, indent=2))


if __name__ == "__main__":
    evaluate_answers()

