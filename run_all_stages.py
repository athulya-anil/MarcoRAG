# run_all_stages.py
"""
One-command runner for Stages 4→8:
- Retrieval → Ground Truth → Retrieval Eval → Answer Gen → Answer Eval

It relies on your existing modules:
  - src.retrieval.run_retrieval_pipeline
  - src.retrieval.ground_truth_gen
  - src.evaluation.retrieval_eval
  - src.answer_generation.answer_gen
  - src.evaluation.llm_answer_eval

Optional:
  - Set RAG_QUERY env var to hint the retrieval step about the query.
    (If your retrieval script doesn't read it yet, it's safely ignored.)
"""

import os
import sys
import subprocess
from pathlib import Path

def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        sys.exit(rc)

def latest_run_dir(base="retrieval_output") -> Path:
    basep = Path(base)
    runs = sorted(basep.glob("run_*"), key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None

if __name__ == "__main__":
    # Stage 4: Retrieval
    run([sys.executable, "-m", "src.retrieval.run_retrieval_pipeline"])

    # Stage 6: Ground Truth
    run([sys.executable, "-m", "src.retrieval.ground_truth_gen"])

    # Stage 7: Retrieval Evaluation
    run([sys.executable, "-m", "src.evaluation.retrieval_eval"])

    # Stage 8A: Answer Generation
    run([sys.executable, "-m", "src.answer_generation.answer_gen"])

    # Stage 8B: Answer Quality Evaluation
    run([sys.executable, "-m", "src.evaluation.llm_answer_eval"])

    out_dir = latest_run_dir()
    if out_dir:
        print(f"\n✅ Finished Stages 4–8.\nArtifacts: {out_dir}")
    else:
        print("\n✅ Finished Stages 4–8.")

