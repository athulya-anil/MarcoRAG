import re
import json
from typing import List, Dict

def preprocess_document(text: str) -> str:
    """Remove boilerplate, legal text, and navigation elements"""
    
    # Patterns to remove
    patterns_to_remove = [
        # Legal/copyright sections
        r'Reverse engineering.*?(?=\n\n|\Z)',
        r'RSA Security Inc\..*?(?=\n\n|\Z)',
        r'This product includes software.*?Apache 2\.0 License.*?(?=\n\n|\Z)',
        
        # Navigation dots and UI elements
        r'^\s*\.{3,}\s*$',
        r'Preface\s*\.{10,}',
        
        # Table of contents entries (long series of dots)
        r'^.*?\.{10,}.*?$',
        
        # Google Cloud console navigation
        r'\[Google Cloud\]\(https://www\.gstatic\.com.*?\)',
        r'Console.*?Sign in',
        r'Contact Us.*?Start free',
        
        # Repeated whitespace
        r'\n{3,}',
    ]
    
    cleaned_text = text
    for pattern in patterns_to_remove:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.MULTILINE)
    
    # Remove chunks that are mostly dots or navigation
    lines = cleaned_text.split('\n')
    filtered_lines = []
    for line in lines:
        dot_ratio = line.count('.') / max(len(line), 1)
        if dot_ratio < 0.5 and len(line.strip()) > 10:
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)


def identify_section_headers(text: str) -> List[Dict]:
    """Identify major section boundaries"""
    
    patterns = [
        # Markdown headers
        (r'^#{1,3}\s+(.+)$', 'header'),
        
        # All caps headers
        (r'^[A-Z\s]{10,}$', 'section'),
        
        # Google Cloud doc patterns
        (r'^##\s+REST Resource:', 'api_section'),
        
        # Tutorial/guide sections
        (r'^\d+\.\s+[A-Z]', 'numbered_section'),
    ]
    
    sections = []
    for i, line in enumerate(text.split('\n')):
        for pattern, section_type in patterns:
            if re.match(pattern, line.strip()):
                sections.append({
                    'line': i,
                    'text': line.strip(),
                    'type': section_type
                })
                break
    
    return sections


def smart_chunk(text: str, 
                target_size: int = 500, 
                max_size: int = 1000,
                min_size: int = 100) -> List[Dict]:
    """
    Chunk text intelligently based on semantic boundaries
    
    Args:
        text: Preprocessed text
        target_size: Target words per chunk
        max_size: Maximum words per chunk
        min_size: Minimum words per chunk
    """
    
    # Split into paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for para in paragraphs:
        para_words = len(para.split())
        
        # Skip very small paragraphs (likely fragments)
        if para_words < 5:
            continue
        
        # If adding this paragraph exceeds max_size, save current chunk
        if current_size + para_words > max_size and current_size >= min_size:
            chunks.append({
                'text': '\n\n'.join(current_chunk),
                'num_words': current_size,
                'num_paragraphs': len(current_chunk)
            })
            current_chunk = [para]
            current_size = para_words
        else:
            current_chunk.append(para)
            current_size += para_words
            
            # If we've hit target size, consider saving
            if current_size >= target_size:
                chunks.append({
                    'text': '\n\n'.join(current_chunk),
                    'num_words': current_size,
                    'num_paragraphs': len(current_chunk)
                })
                current_chunk = []
                current_size = 0
    
    # Add remaining content
    if current_chunk and current_size >= min_size:
        chunks.append({
            'text': '\n\n'.join(current_chunk),
            'num_words': current_size,
            'num_paragraphs': len(current_chunk)
        })
    
    return chunks


def add_context_headers(chunks: List[Dict], original_text: str) -> List[Dict]:
    """Add contextual headers to each chunk for better retrieval"""
    
    # Find major section headers in original text
    current_section = "General"
    
    enhanced_chunks = []
    for chunk in chunks:
        # Find the section this chunk belongs to
        chunk_start = original_text.find(chunk['text'][:50])
        
        # Look backwards for the nearest header
        if chunk_start != -1:
            preceding_text = original_text[:chunk_start]
            headers = re.findall(r'^#{1,3}\s+(.+)$', preceding_text, re.MULTILINE)
            if headers:
                current_section = headers[-1]
        
        enhanced_chunks.append({
            **chunk,
            'section': current_section,
            'metadata': {
                'section': current_section,
                'word_count': chunk['num_words']
            }
        })
    
    return enhanced_chunks


def filter_low_quality_chunks(chunks: List[Dict]) -> List[Dict]:
    """Remove chunks that are likely low quality"""
    
    quality_chunks = []
    
    for chunk in chunks:
        text = chunk['text']
        
        # Skip if mostly code/symbols
        alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
        if alpha_ratio < 0.5:
            continue
        
        # Skip if very repetitive
        words = text.lower().split()
        unique_ratio = len(set(words)) / max(len(words), 1)
        if unique_ratio < 0.3:
            continue
        
        # Skip if mostly URLs or paths
        if text.count('http') > 5 or text.count('://') > 5:
            continue
        
        quality_chunks.append(chunk)
    
    return quality_chunks


def process_document(raw_text: str) -> List[Dict]:
    """Main processing pipeline"""
    
    print("1. Preprocessing document...")
    cleaned_text = preprocess_document(raw_text)
    
    print("2. Creating semantic chunks...")
    chunks = smart_chunk(cleaned_text, target_size=500, max_size=1000, min_size=100)
    
    print("3. Adding contextual headers...")
    chunks = add_context_headers(chunks, cleaned_text)
    
    print("4. Filtering low-quality chunks...")
    chunks = filter_low_quality_chunks(chunks)
    
    print(f"5. Final output: {len(chunks)} chunks")
    
    # Add IDs
    for i, chunk in enumerate(chunks):
        chunk['chunk_id'] = f"vertex_ai_improved_{i}"
    
    return chunks


def load_from_existing_json(json_file: str) -> str:
    """Load text from your existing JSON format"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Concatenate all the text from chunks
    all_text = []
    for chunk in data:
        if 'text' in chunk:
            all_text.append(chunk['text'])
    
    return '\n\n'.join(all_text)


# Example usage
if __name__ == "__main__":
    # OPTION 1: If you have a plain text file
    print("Loading document...")
    try:
        with open('vertex_ai_corpus.txt', 'r', encoding='utf-8') as f:
            raw_text = f.read()
        print(f"✓ Loaded {len(raw_text)} characters from vertex_ai_corpus.txt")
    except FileNotFoundError:
        print("❌ Error: vertex_ai_corpus.txt not found!")
        print("Please make sure the file is in the same folder as this script.")
        exit(1)
    
    # OPTION 2: Or if loading from your old JSON format, uncomment below:
    # raw_text = load_from_existing_json('vertex_ai_corpus_cleaned_semantic_chunks.json')
    
    # Process
    improved_chunks = process_document(raw_text)
    
    # Save
    output_file = 'vertex_ai_corpus_improved.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(improved_chunks, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved {len(improved_chunks)} chunks to {output_file}")
    
    # Print statistics
    print("\n=== Statistics ===")
    total_words = sum(c['num_words'] for c in improved_chunks)
    avg_words = total_words / len(improved_chunks) if improved_chunks else 0
    print(f"Total chunks: {len(improved_chunks)}")
    print(f"Total words: {total_words}")
    print(f"Average words per chunk: {avg_words:.0f}")
    
    # Print first 3 chunks as samples
    print("\n=== Sample Chunks ===")
    for i, sample in enumerate(improved_chunks[:3]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Section: {sample.get('section', 'N/A')}")
        print(f"Words: {sample['num_words']}")
        print(f"Preview: {sample['text'][:150]}...")
        print("---")
    
    print(f"\n✅ Done! Check {output_file} for the full results.")
