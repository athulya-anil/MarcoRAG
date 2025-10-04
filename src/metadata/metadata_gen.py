# src/metadata/metadata_gen.py
"""
Metadata Generation Module (Enhanced Stage 3.5)
-----------------------------------------------
Adds semantic enrichment to chunked JSON using Groq LLaMA-3.
Performs lightweight pre-classification (title / procedure / reference / misc)
before calling the LLM for richer, non-repetitive metadata.

Usage:
    python src/metadata/metadata_gen.py
"""

import os
import re
import json
import time
from tqdm import tqdm
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------- Helper Utilities ---------------- #

def _clean_json_string(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?", "", s, flags=re.IGNORECASE)
    s = re.sub(r"```$", "", s)
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s.strip()

def _safe_json_parse(s: str):
    try:
        return json.loads(s)
    except Exception:
        cleaned = _clean_json_string(s)
        try:
            return json.loads(cleaned)
        except Exception:
            return None

def _guess_pre_category(text: str) -> str:
    """Heuristic pre-classification before LLM call."""
    t = text.lower().strip()
    if len(t) < 20:
        return "Ignored"
    if t.startswith("http"):
        return "Ignored"
    if re.match(r"^\d+/\d+$", t) or re.match(r"^\d{1,2}/\d{1,2}/\d{4}", t):
        return "Ignored"
    if "how to" in t or "steps" in t:
        return "Procedure"
    if "code" in t or "example" in t:
        return "Example"
    if "api" in t or "reference" in t or "docs" in t:
        return "Reference"
    if "learn more" in t or "get started" in t:
        return "CTA"
    if re.match(r"^[A-Z][A-Za-z0-9\s\-]+$", text) and len(t.split()) < 6:
        return "Title"
    return "Concept"

# ---------------- Core LLM Call ---------------- #

def generate_metadata_with_llm(text: str, pre_category: str = "", model="llama-3.1-8b-instant"):
    """Generate metadata from text with context-aware prompt."""
    if pre_category == "Ignored" or not text.strip():
        return {"summary": "Skipped trivial chunk", "entities": [], "keywords": [], "category": "Ignored"}

    prompt = f"""
You are an AI metadata generator for documentation chunks.
Analyze the following text and produce a concise JSON with:
- summary: 1–2 sentences summarizing meaning and intent
- entities: key products, APIs, or technologies mentioned
- keywords: 4–8 domain-relevant terms
- category: choose one best label from [Concept, Procedure, Code, Example, Architecture, Reference, CTA, Misc]

The chunk appears to be a **{pre_category}** type.
Text:
{text}

Return valid JSON only.
"""

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a metadata extraction assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=300,
            )
            content = (response.choices[0].message.content or "").strip()
            parsed = _safe_json_parse(content)
            if isinstance(parsed, dict):
                return parsed
        except Exception as e:
            print(f"⚠️ Retry {attempt+1}/3: {e}")
            time.sleep(1)
    return {"summary": "", "entities": [], "keywords": [], "category": pre_category or "Unknown"}

# ---------------- File-Level Enrichment ---------------- #

def enrich_chunk_file(input_path: str, output_path: str, fast_mode=True):
    with open(input_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    enriched = []
    print(f"✨ Processing {len(chunks)} chunks from {os.path.basename(input_path)}")

    for chunk in tqdm(chunks, desc="Generating metadata", ncols=80):
        text = chunk.get("text", "")
        metadata = chunk.get("metadata", {})
        pre_category = _guess_pre_category(text)

        # Fast skip
        if pre_category == "Ignored" and fast_mode:
            enriched.append({
                "chunk_id": chunk["chunk_id"],
                "text": text,
                "metadata": {
                    **metadata,
                    "summary": "Skipped trivial chunk",
                    "entities": [],
                    "keywords": [],
                    "category": "Ignored",
                },
            })
            continue

        llm_metadata = generate_metadata_with_llm(text, pre_category)
        metadata.update(llm_metadata)
        enriched.append({
            "chunk_id": chunk["chunk_id"],
            "text": text,
            "metadata": metadata,
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2)

    print(f"✅ Saved enriched metadata → {output_path}")

# ---------------- Runner ---------------- #

def run_metadata_generation(input_dir="chunk_output", output_dir="metadata_output"):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
    if not files:
        print("❌ No JSON files found.")
        return
    for fname in files:
        enrich_chunk_file(
            os.path.join(input_dir, fname),
            os.path.join(output_dir, fname.replace("_chunks", "_metadata"))
        )
    print("\n🎉 Metadata generation completed for all files!")

if __name__ == "__main__":
    run_metadata_generation()
