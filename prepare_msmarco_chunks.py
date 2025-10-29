"""
Prepare MS MARCO passages as chunks.
Since MS MARCO passages are already short, we'll treat each passage as one chunk.
"""

import json
import re
from pathlib import Path

def prepare_chunks():
    """Parse MS MARCO passages and create chunk files."""

    input_file = Path("input_files/msmarco_passages.txt")
    output_dir = Path("chunk_output")
    output_dir.mkdir(exist_ok=True)

    print(f"📖 Reading passages from {input_file}...")

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by passage markers
    passages = re.split(r'\n---\n', content)

    chunks = []
    for passage in passages:
        passage = passage.strip()
        if not passage:
            continue

        # Extract passage ID and text
        match = re.match(r'\[([^\]]+)\]\n(.+)', passage, re.DOTALL)
        if match:
            passage_id = match.group(1)
            text = match.group(2).strip()

            chunks.append({
                "chunk_id": passage_id,
                "text": text,
                "metadata": {
                    "source": "ms_marco",
                    "original_id": passage_id
                }
            })

    # Save chunks
    output_file = output_dir / "msmarco_passages_chunks.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2)

    print(f"✅ Created {len(chunks)} chunks")
    print(f"💾 Saved to {output_file}")

    return chunks

if __name__ == "__main__":
    prepare_chunks()
