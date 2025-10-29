# MarcoRAG 

**An end-to-end Retrieval-Augmented Generation (RAG) pipeline** with proper evaluation using the MS MARCO dataset.

Unlike traditional RAG systems that use circular self-validation, MarcoRAG evaluates retrieval quality against **real human-labeled ground truth** from MS MARCO. It chunks documents, enriches them with metadata, builds embeddings, retrieves relevant context, generates answers with an LLM (Groq), and provides trustworthy evaluation metrics - all with reproducible run artifacts and a simple Streamlit UI.

---

## Table of Contents

- [Introduction](#introduction)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Quickstart](#quickstart)
- [Project Components](#project-components)
- [One-Command Run](#one-command-run-stages-48)
- [Streamlit Demo](#streamlit-demo)
- [Outputs & Run Layout](#outputs--run-layout)
- [Example Results](#example-results)
- [Repository Structure](#repository-structure)
- [Configuration](#configuration)
- [Notes & Limitations](#notes--limitations)
- [Roadmap](#roadmap)
- [License](#license)
- [Citation](#citation)

---

## Introduction

Retrieval-Augmented Generation (RAG) improves LLM answers by **retrieving** relevant context first and then **generating** a response grounded in that context.

**MarcoRAG** implements a practical, modular RAG pipeline with a key differentiator: **real evaluation using human-labeled ground truth from MS MARCO**.

Key features:
- Breaks documents into meaningful chunks
- Enriches chunks with semantic metadata
- Builds embeddings for fast similarity search
- Retrieves and (optionally) reranks context
- Generates answers with Groq LLM
- Evaluates against **MS MARCO human annotations** (not circular self-validation)
- Achieves **91% Recall@5** on MS MARCO dataset

---

## Architecture

```
Docs → Chunking → Metadata → Embeddings → Retrieval (+Reranker)
  → Ground Truth → Retrieval Eval → Answer Gen (Groq) → Answer Eval
```

_All stages write timestamped artifacts to `retrieval_output/run_<timestamp>/`._

---

## Requirements

- **Python**: 3.11 recommended
- **Install dependencies**:
  ```bash
  python3.11 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```

- **Environment**: Create a `.env` file in the project root with:
  ```
  GROQ_API_KEY=your_key
  # Optional override (defaults exist in code)
  # GROQ_MODEL=llama-3.3-70b-versatile
  ```

---

## Quickstart

1. **Prepare inputs** in `input_files/` (plain text preferred)
2. **(Optional)** Generate chunks, metadata, embeddings via provided scripts
3. **Run Stages 4→8** (see [One-Command Run](#one-command-run-stages-48))
4. **Inspect outputs** in the `retrieval_output/` directory
5. **(Optional)** Launch the Streamlit app and ask questions live

---

## Project Components

1. **Chunking** - Splits documents into coherent segments (`src/chunking/`)
2. **Metadata Enrichment** - Adds summaries, keywords, entities via LLM (`src/metadata/`)
3. **Embeddings** - Vectorizes content for semantic search (`src/embeddings/`)
4. **Retrieval** - Retrieves top-K chunks with grounding metrics (`src/retrieval/`)
5. **Ground Truth** - Builds pseudo ground truth with cross-encoder reranker
6. **Retrieval Evaluation** - Computes Precision@K, Recall@K, MRR, NDCG
7. **Answer Generation** - Generates answers via Groq LLM
8. **Answer Quality Evaluation** - Scores faithfulness, completeness, hallucination

---

## One-Command Run (Stages 4→8)

> Assumes your metadata and inputs are already prepared (as in this repo's example).
> The script sequentially runs: Retrieval → GT → Retrieval Eval → Answer Gen → Answer Eval.

```bash
python run_all_stages.py
```

Artifacts will appear under `retrieval_output/run_<timestamp>/`.

---

## Streamlit Demo

Run a lightweight UI to ask questions:

```bash
streamlit run app.py
```

The app calls your pipeline function and displays the generated answer.

---

## Outputs & Run Layout

A typical run produces:

```
retrieval_output/
  run_YYYY-MM-DD_HH-MM-SS/
    retrieval_results.json
    metrics_overview.json
    ground_truth/
      gt.json
    evaluation/
      metrics.json
    answers/
      answers.json
      answer_eval.json
```

---

## Example Results

**MS MARCO Evaluation Results** (33 queries, 507 passages with human ground truth):

**Retrieval Metrics**

- **Recall@5: 0.91** (91% success rate - finds relevant passage in top-5)
- **Precision@5: 0.18** (18%, near-optimal for single-passage queries)
- **NDCG@5: 0.70** (70%, relevant passages ranked highly)

**What This Means:**
- ✅ Successfully retrieves the correct passage for **30 out of 33 queries**
- ✅ Evaluated against **real human annotations** from MS MARCO
- ✅ Performance exceeds typical academic RAG benchmarks (60-85% Recall@5)
- ✅ No circular validation - these are trustworthy metrics

**Sample Queries Answered:**
- "what was the immediate impact of the success of the manhattan project?"
- "why did stalin want control of eastern europe"
- "are whiskers on cats used for balance"
- "what does folic acid do"

---

## Repository Structure

```
src/
  chunking/                   # Stage 1
  metadata/                   # Stage 2 (+ basic eval)
  embeddings/                 # Stage 3 (prefix embedder implemented)
  retrieval/                  # Stages 4 & 5 (GT)
    run_retrieval_pipeline.py
    ground_truth_gen.py
  evaluation/                 # Stages 6 & 8
    retrieval_eval.py
    llm_answer_eval.py
  answer_generation/          # Stage 7
    answer_gen.py
app.py                        # Streamlit UI
run_all_stages.py             # Orchestrates Stages 4→8
input_files/                  # Sample inputs
chunk_output/                 # Chunking artifacts
metadata_output/              # Metadata artifacts
embeddings_output/            # Embedding artifacts
retrieval_output/             # Run-specific outputs
```

---

## Configuration

- **Environment**: `.env` with `GROQ_API_KEY` (required for answer generation)
- **Models**: Groq model default set in code; update as needed if deprecations occur
- **Python**: 3.11 recommended. Dependencies pinned in `requirements.txt`

---

## Notes & Limitations

- Evaluation uses MS MARCO dataset with 507 passages and 33 queries with human-labeled relevance
- Run `python run_msmarco_evaluation.py` to reproduce evaluation results
- Embedding bug fix: Ensure `prefix_embedder.py` handles both 'text' and 'content' fields
- Reranking improves precision but adds latency
- Current answer evaluation uses similarity-based scoring; entity-level scoring can be added

**Why MS MARCO?**
- Provides real human relevance judgments (not LLM-generated)
- Avoids circular validation (where ground truth is derived from retrieval results)
- Industry-standard benchmark for retrieval systems

---

## License

MIT
