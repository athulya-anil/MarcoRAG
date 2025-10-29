# app.py
import os
import json
from pathlib import Path
import streamlit as st
import subprocess
import sys

st.set_page_config(page_title="MarcoRAG", page_icon="🔍")
st.title("MarcoRAG 🔍")
st.write("Ask a question about your docs and see the end-to-end RAG pipeline run (Stages 4→8).")

# --- Helpers --- #
def run_pipeline():
    """Run all pipeline stages with proper error capture."""
    cmds = [
        [sys.executable, "-m", "src.retrieval.run_retrieval_pipeline"],
        [sys.executable, "-m", "src.retrieval.ground_truth_gen"],
        [sys.executable, "-m", "src.evaluation.retrieval_eval"],
        [sys.executable, "-m", "src.answer_generation.answer_gen"],
        [sys.executable, "-m", "src.evaluation.llm_answer_eval"],
    ]
    for cmd in cmds:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error_msg = f"Command failed: {' '.join(cmd)}\n"
            if result.stderr:
                error_msg += f"Error output: {result.stderr}"
            raise RuntimeError(error_msg)

def latest_run_dir(base="retrieval_output") -> Path:
    runs = sorted(Path(base).glob("run_*"), key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None

def try_load_json(p: Path):
    if p and p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None

def extract_answer(answers_json: dict):
    """Handle common shapes. Your current structure shows a 'query_1' entry."""
    if not isinstance(answers_json, dict):
        return None, None, None
    # Try simple shape
    if "answer" in answers_json and "context" in answers_json:
        return answers_json.get("answer"), answers_json.get("context"), answers_json.get("query")
    # Try keyed entries: {"query_1": {...}}
    for _, v in answers_json.items():
        if isinstance(v, dict) and "answer" in v:
            return v.get("answer"), v.get("context"), v.get("query")
    return None, None, None

# --- UI --- #
with st.form("rag_form"):
    query = st.text_area("Enter your question:", height=100, placeholder="e.g., What is Retrieval-Augmented Generation?")
    run_btn = st.form_submit_button("Run Pipeline (Stages 4→8)")

if run_btn:
    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()

    # Hint the retrieval step about the query via env var (safe even if ignored)
    os.environ["RAG_QUERY"] = query.strip()

    with st.spinner("Running retrieval → GT → eval → answer → answer-eval..."):
        try:
            run_pipeline()
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.stop()

    # Load latest artifacts
    run_dir = latest_run_dir()
    if not run_dir:
        st.error("No run folder found under retrieval_output/.")
        st.stop()

    st.success(f"Completed. Loaded artifacts from: {run_dir}")

    # Answers
    answers_path = run_dir / "answers" / "answers.json"
    answers_json = try_load_json(answers_path)
    answer, context_list, q_used = extract_answer(answers_json or {})

    st.subheader("💡 Answer")
    if answer:
        st.write(answer)
    else:
        st.write("_No answer found in answers.json_")

    # Show top context if present
    if context_list and isinstance(context_list, list):
        st.subheader("📚 Top Context")
        for i, c in enumerate(context_list[:5], 1):
            with st.expander(f"Context #{i}"):
                st.write(c)

    # Retrieval metrics
    metrics_path = run_dir / "evaluation" / "metrics.json"
    metrics_json = try_load_json(metrics_path)
    if metrics_json:
        st.subheader("📈 Retrieval Metrics")
        st.json(metrics_json)

    # Answer evaluation
    answer_eval_path = run_dir / "answers" / "answer_eval.json"
    answer_eval_json = try_load_json(answer_eval_path)
    if answer_eval_json:
        st.subheader("🧪 Answer Quality")
        st.json(answer_eval_json)

    # Raw files quick access
    with st.expander("Raw files"):
        st.write(f"- `{answers_path}`")
        st.write(f"- `{metrics_path}`")
        st.write(f"- `{answer_eval_path}`")

st.caption("Tip: Better, cleaner input docs generally yield better retrieval and answer metrics.")
