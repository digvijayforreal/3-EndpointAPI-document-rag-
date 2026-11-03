
import os
import requests

from backend_v2.database.mongodb import (
    documents_collection,
    chunks_collection
)

JINA_URL = "https://api.jina.ai/v1/embeddings"
JINA_MODEL = "jina-embeddings-v3"
BATCH_SIZE = 10
DIMENSIONS = 1024


def generate_embeddings(texts):
    api_key = os.getenv("JINA_API_KEY")

    if not api_key:
        raise RuntimeError("JINA_API_KEY missing in .env")

    response = requests.post(
        JINA_URL,
        headers={
            "Authorization": f"Bearer {api_key}"
        },
        json={
            "model": JINA_MODEL,
            "task": "retrieval.passage",
            "dimensions": DIMENSIONS,
            "input": texts
        },
        timeout=120
    )

    response.raise_for_status()
    data = response.json()["data"]

    # Preserve input order
    data = sorted(data, key=lambda item: item["index"])

    vectors = [item["embedding"] for item in data]

    if len(vectors) != len(texts):
        raise ValueError("Embedding count mismatch")

    if any(len(vector) != DIMENSIONS for vector in vectors):
        raise ValueError("Unexpected embedding dimensions")

    return vectors


def embed_document(document_id):
    document = documents_collection.find_one({
        "document_id": document_id
    })

    if not document:
        raise ValueError("Document not found")

    chunks = list(
        chunks_collection.find({
            "document_id": document_id,
            "embedding_status": {"$ne": "embedded"}
        }).sort("chunk_id", 1)
    )

    if not chunks:
        return {
            "message": "All chunks are already embedded",
            "document_id": document_id,
            "total_embedded": 0,
            "embedding_status": "completed"
        }

    total_embedded = 0

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]

        texts = [chunk["text"] for chunk in batch]
        vectors = generate_embeddings(texts)

        for chunk, vector in zip(batch, vectors):
            chunks_collection.update_one(
                {"_id": chunk["_id"]},
                {
                    "$set": {
                        "embedding_jina": vector,
                        "embedding_model": JINA_MODEL,
                        "embedding_status": "embedded"
                    }
                }
            )

            total_embedded += 1

    remaining = chunks_collection.count_documents({
        "document_id": document_id,
        "embedding_status": {"$ne": "embedded"}
    })

    status = "completed" if remaining == 0 else "pending"

    documents_collection.update_one(
        {"document_id": document_id},
        {"$set": {"embedding_status": status}}
    )

    return {
        "message": "Embeddings generated successfully",
        "document_id": document_id,
        "total_embedded": total_embedded,
        "embedding_status": status
    }
