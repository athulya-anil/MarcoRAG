# Vertex RAG Engine 🔍

**An end-to-end Retrieval-Augmented Generation (RAG) pipeline** for technical documentation with a focus on Google Cloud/Vertex AI materials.

It chunks documents, enriches them with metadata, builds embeddings, retrieves relevant context, generates answers with an LLM (Groq), and evaluates answer quality — all with reproducible run artifacts and a simple Streamlit UI.

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

**Vertex RAG Engine** implements a practical, modular RAG pipeline:

- Breaks documents into meaningful chunks
- Enriches chunks with semantic metadata
- Builds embeddings for fast similarity search
- Retrieves and (optionally) reranks context
- Generates answers with Groq LLM
- Evaluates both **retrieval quality** and **answer quality**

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

From a recent local run (K=5):

**Retrieval Metrics**

- Precision@5: **0.60**
- Recall@5: **1.00**
- MRR: **0.50**
- NDCG@5: **0.68**

**Answer Quality (average)**

- Faithfulness: **0.117**
- Completeness: **0.491**
- Hallucination: **0.883**

> These numbers were produced on a small, simple context set and are expected to improve with **cleaner, richer source documents**.

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

- Small demo corpus → metrics vary; better documents yield better retrieval and answer scores
- Reranking improves precision but adds latency
- Current answer evaluation uses similarity-based scoring; entity-level scoring can be added

---

## License

MIT