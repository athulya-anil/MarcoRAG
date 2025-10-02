#RAG pipeline
#version 1.0

import os
import argparse
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()

# Init Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def load_documents(input_dir):
    """Load all .txt files from input_dir"""
    docs = {}
    for fname in os.listdir(input_dir):
        if fname.endswith(".txt"):
            with open(os.path.join(input_dir, fname), "r", encoding="utf-8") as f:
                docs[fname] = f.read()
    return docs


def chunk_text(text, chunk_size=800, overlap=100):
    """Split text into overlapping chunks for better retrieval"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def embed_documents(docs, model):
    """Generate embeddings for document chunks"""
    embeddings = []
    for fname, text in docs.items():
        for i, chunk in enumerate(chunk_text(text)):
            emb = model.encode(chunk)
            embeddings.append({
                "fname": fname,
                "chunk_id": i,
                "text": chunk,
                "embedding": emb
            })
    return embeddings


def retrieve(query, embeddings, model, top_k=3):
    """Retrieve top_k most relevant chunks for a query"""
    query_vec = model.encode(query).reshape(1, -1)
    scored = []
    for item in embeddings:
        score = cosine_similarity(query_vec, item["embedding"].reshape(1, -1))[0][0]
        scored.append((item, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def generate_answer(query, retrieved_chunks):
    """Call Groq LLM to generate an answer"""
    context = "\n\n".join(
        [f"From {c['fname']} (chunk {c['chunk_id']}):\n{c['text']}" 
         for c, _ in retrieved_chunks]
    )
    prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {query}\nAnswer:"

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # or "mixtral-8x7b" for faster/cheaper
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer only using the provided context."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=400,
    )

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="input_files", help="Path to input text files")
    parser.add_argument("--query", type=str, required=True, help="Your query to test retrieval+generation")
    args = parser.parse_args()

    print("📂 Loading documents...")
    docs = load_documents(args.input_dir)

    print("🔎 Embedding documents...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_documents(docs, model)

    print("📑 Retrieving top chunks...")
    retrieved = retrieve(args.query, embeddings, model)

    print("\n❓ Query:", args.query)
    print("\n📌 Retrieved from:")
    for item, score in retrieved:
        print(f"- {item['fname']} (chunk {item['chunk_id']}) [score={score:.4f}]")

    answer = generate_answer(args.query, retrieved)
    print("\n💡 Answer:\n", answer)
def run_rag(query: str):
    docs = load_documents("input_files")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_documents(docs, model)
    retrieved = retrieve(query, embeddings, model)
    answer = generate_answer(query, retrieved)
    return answer

