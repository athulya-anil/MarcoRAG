# Vertex RAG Engine 🔍

End-to-end Retrieval-Augmented Generation (RAG) engine for Google Vertex AI documentation — embeddings, retrieval, and Groq-powered answer generation.

---

## Overview
This project implements a simple RAG pipeline:
1. Load Google Vertex AI documentation from `input_files/`.
2. Generate embeddings with [SentenceTransformers](https://www.sbert.net/).
3. Retrieve the most relevant chunks using FAISS similarity search.
4. Use the [Groq LLM API](https://groq.com/) to generate answers based on retrieved context.

## 🚀 Project Stages

| Stage | Description | Status |
|:------|:-------------|:--------|
| **1** | Data Ingestion and Preprocessing | ✅ Completed |
| **2** | Chunking Factory (multi-chunk generation) | ✅ Completed |
| **3** | Metadata Generation | ✅ Completed |
| **4** | Embedding Model Setup | ✅ Completed |
| **5** | **Embedding Generation Pipeline** | 🟢 **Completed (Merged via PR)** |
| **6** | Retriever Evaluation and Similarity Search | 🔄 In Progress |
| **7** | RAG Response Generation and UI Integration | ⏳ Pending |