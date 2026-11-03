
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def create_chunks(sections: list[dict]) -> list[dict]:
    chunks = []

    for section in sections:
        text = section["text"]
        page = section["page"]

        start = 0

        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "chunk_id": len(chunks) + 1,
                    "text": chunk_text,
                    "page": page
                })

            if end == len(text):
                break

            start = end - CHUNK_OVERLAP

    return chunks
