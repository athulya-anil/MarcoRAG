# src/metadata/metadata_gen.py
"""
Metadata Generation Module (Stage 3)
------------------------------------
Enriches chunked JSON files with semantic metadata such as summaries,
entities, keywords, and content categories using Groq's LLaMA-3 model.

Usage:
    python run_metadata.py --input_dir chunk_output --output_dir metadata_output
"""

import os
import json
import time
from tqdm import tqdm
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_metadata_with_llm(text: str, model: str = "llama-3.1-8b-instant", retries: int = 3):
    """
    Calls the LLM to generate structured metadata for a given text chunk.
    """
    prompt = f"""
    You are an expert AI metadata generator.
    Given the following text, extract structured metadata in JSON with these keys:
    - summary: 1-2 sentence summary of the content
    - entities: main named entities (people, products, technologies)
    - keywords: 4-8 key terms capturing main topics
    - category: one-word category label (e.g., Architecture, Code, Concept, Procedure, Example)

    Text:
    {text}

    Return ONLY valid JSON.
    """

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a metadata extraction assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=300,
            )

            content = response.choices[0].message.content.strip()

            # Try parsing JSON output from the model
            metadata = json.loads(content)
            return metadata

        except Exception as e:
            print(f"⚠️ Retry {attempt+1}/{retries} due to error: {e}")
            time.sleep(2)

    print("❌ Failed to generate metadata after multiple retries.")
    return {
        "summary": "",
        "entities": [],
        "keywords": [],
        "category": "Unknown"
    }


def enrich_chunk_file(input_path: str, output_path: str):
    """
    Reads one chunk JSON file and generates enriched metadata for each chunk.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    enriched = []
    print(f"✨ Processing {len(chunks)} chunks from {os.path.basename(input_path)}")

    for chunk in tqdm(chunks, desc="Generating metadata", ncols=80):
        text = chunk.get("text", "")
        metadata = chunk.get("metadata", {})

        # Generate new metadata fields
        llm_metadata = generate_metadata_with_llm(text)
        metadata.update(llm_metadata)

        enriched.append({
            "chunk_id": chunk["chunk_id"],
            "text": text,
            "metadata": metadata
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2)

    print(f"✅ Saved enriched metadata → {output_path}")


def run_metadata_generation(input_dir: str = "chunk_output", output_dir: str = "metadata_output"):
    """
    Runs metadata generation for all chunked JSON files in the input directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]

    if not json_files:
        print("❌ No chunk JSON files found in input directory.")
        return

    for fname in json_files:
        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname.replace("_chunks", "_metadata"))
        enrich_chunk_file(input_path, output_path)

    print("\n🎉 Metadata generation completed for all files!")


if __name__ == "__main__":
    run_metadata_generation()

