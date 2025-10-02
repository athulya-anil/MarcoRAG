# Vertex RAG Engine 🔍

End-to-end Retrieval-Augmented Generation (RAG) engine for Google Vertex AI documentation — embeddings, retrieval, and Groq-powered answer generation.

---

## Overview
This project implements a simple RAG pipeline:
1. Load Google Vertex AI documentation from `input_files/`.
2. Generate embeddings with [SentenceTransformers](https://www.sbert.net/).
3. Retrieve the most relevant chunks using FAISS similarity search.
4. Use the [Groq LLM API](https://groq.com/) to generate answers based on retrieved context.
