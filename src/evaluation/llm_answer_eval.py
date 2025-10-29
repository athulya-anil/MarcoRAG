import os, json, argparse
import numpy as np
from pathlib import Path
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def get_latest_run_id():
    """Get the most recent run directory."""
    runs = sorted(Path("retrieval_output").glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0].name if runs else None

def evaluate_answers():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=False, default=None, help="Run ID to evaluate (defaults to latest)")
    args = parser.parse_args()

    run_id = args.run_id or get_latest_run_id()
    if not run_id:
        print("❌ No runs found in retrieval_output/")
        return

    run_dir = os.path.join("retrieval_output", run_id)
    answers_path = os.path.join(run_dir, "answers", "answers.json")
    gt_path = os.path.join(run_dir, "ground_truth", "gt.json")
    eval_dir = os.path.join(run_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    metrics_path = os.path.join(eval_dir, "answer_metrics.json")

    print(f"📄 Answers: {answers_path}")
    print(f"📄 Ground Truth: {gt_path}")

    with open(answers_path, "r") as f:
        answers = json.load(f)
    with open(gt_path, "r") as f:
        ground_truth = json.load(f)

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    bleu_scores, rouge_scores, bert_scores, faith_scores = [], [], [], []

    print("\n🧠 Evaluating answers...\n")
    for qid, entry in answers.items():
        query = entry.get("query", "")
        gen_answer = entry.get("answer", "").strip()
        gt_answer = ground_truth.get(qid, {}).get("answer", "").strip()

        # ✅ Handle context safely (use used_docs if context missing)
        ctx_list = entry.get("context") or entry.get("used_docs") or []
        context = " ".join(ctx_list) if isinstance(ctx_list, list) else str(ctx_list)

        # BLEU
        smooth = SmoothingFunction().method1
        bleu = sentence_bleu([gt_answer.split()], gen_answer.split(), smoothing_function=smooth)
        bleu_scores.append(bleu)

        # ROUGE-L
        rougeL = rouge.score(gt_answer, gen_answer)["rougeL"].fmeasure
        rouge_scores.append(rougeL)

        # BERTScore
        P, R, F1 = bert_score([gen_answer], [gt_answer], lang="en", verbose=False)
        bert_scores.append(float(F1.mean()))

        # Faithfulness (answer vs context)
        emb = embed_model.encode([gen_answer, context])
        faith = float(cosine_similarity([emb[0]], [emb[1]])[0][0])
        faith_scores.append(faith)

        print(f"✅ {qid} | BLEU={bleu:.3f} ROUGE-L={rougeL:.3f} BERT={F1.mean():.3f} Faith={faith:.3f}")

    metrics = {
        "BLEU": float(np.mean(bleu_scores)),
        "ROUGE-L": float(np.mean(rouge_scores)),
        "BERTScore": float(np.mean(bert_scores)),
        "Faithfulness": float(np.mean(faith_scores)),
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n✅ Evaluation complete.")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    evaluate_answers()
