def chunk_text(text: str, size: int = 512, overlap: int = 64):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks
