from typing import List

def chunk_text(text: str, max_length: int = 700, overlap: int = 150) -> List[str]:
    """Split text into word-based chunks with overlap."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+max_length]
        chunks.append(" ".join(chunk))
        i += max_length - overlap
    return chunks
