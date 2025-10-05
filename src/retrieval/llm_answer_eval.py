"""
LLM Answer Evaluation
---------------------
Evaluates LLM-generated answers for faithfulness and groundedness
against retrieved evidence.
"""

def evaluate_answer(answer: str, evidence_chunks):
    evidence_text = " ".join(c["text"].lower() for c in evidence_chunks)
    overlap = len(set(answer.lower().split()) & set(evidence_text.split()))
    groundedness = overlap / max(len(answer.split()), 1)
    return {"faithfulness": min(1.0, groundedness * 2), "groundedness": groundedness}

