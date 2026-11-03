
def chunk_text(pages, chunk_size=800, overlap=150):
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Invalid chunk size or overlap")

    chunks = []
    step = chunk_size - overlap

    for page in pages:
        text = page["text"]

        for start in range(0, len(text), step):
            content = text[start:start + chunk_size].strip()

            if content:
                chunks.append({
                    "page": page["page"],
                    "chunk_id": len(chunks),
                    "text": content
                })

            if start + chunk_size >= len(text):
                break

    return chunks
