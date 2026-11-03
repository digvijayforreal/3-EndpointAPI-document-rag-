"""Multiple-PDF ingestion and document-scoped RAG for the existing project."""
import hashlib
import io
import os
from pathlib import Path
from urllib.parse import quote_plus

import requests
from groq import Groq
from pymongo import MongoClient, ReplaceOne
from pypdf import PdfReader

from rag_chat import (DIMENSIONS, INDEX_NAME, MODEL, REFUSAL, embed_question,
                      generate_answer, validate_answer)

MAX_BYTES = 8 * 1024 * 1024
MAX_DOCS = 20  # Demo workspace limit; adjust to your needs.
BATCH_SIZE = 12


def connect_db():
    user = quote_plus(os.environ["MONGODB_USERNAME"])
    password = quote_plus(os.environ["MONGODB_PASSWORD"])
    host = os.environ["MONGODB_HOST"]
    uri = f"mongodb+srv://{user}:{password}@{host}/?appName=Cluster0"
    client = MongoClient(uri, serverSelectionTimeoutMS=10000)
    client.admin.command("ping")
    collection = client[os.getenv("DB_NAME", "rag_database")][os.getenv("COLLECTION_NAME", "documents")]
    return client, collection


def extract_pages(data: bytes) -> list[dict]:
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs are not supported. Upload an unlocked text-based PDF.")
    pages = []
    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page": i, "text": text.strip()})
    if not pages:
        raise ValueError("No selectable text found. Scanned/image-only PDFs need OCR.")
    return pages


def make_chunks(pages: list[dict], size: int = 800, overlap: int = 150) -> list[dict]:
    if not (size > 0 and 0 <= overlap < size):
        raise ValueError("Invalid chunk size or overlap")
    chunks = []
    for page in pages:
        text = page["text"]
        step = size - overlap
        for start in range(0, len(text), step):
            piece = text[start:start + size].strip()
            if piece:
                chunks.append({"page": page["page"], "chunk_id": len(chunks), "text": piece})
            if start + size >= len(text):
                break
    return chunks


def embed_passages(chunks: list[dict]) -> list[list[float]]:
    vectors = []
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        response = requests.post(
            "https://api.jina.ai/v1/embeddings",
            headers={"Authorization": f"Bearer {os.environ['JINA_API_KEY']}"},
            json={"model": "jina-embeddings-v3", "task": "retrieval.passage",
                  "dimensions": DIMENSIONS, "input": [c["text"] for c in batch]},
            timeout=120,
        )
        response.raise_for_status()
        items = sorted(response.json()["data"], key=lambda x: x["index"])
        if len(items) != len(batch):
            raise ValueError("Jina returned an unexpected number of embeddings")
        for item in items:
            vec = item["embedding"]
            if len(vec) != DIMENSIONS:
                raise ValueError("Jina embedding dimensions do not match MongoDB index")
            vectors.append(vec)
    return vectors


def list_documents(collection) -> list[dict]:
    # One entry per distinct document ID, including the previous sample_pdf.
    pipeline = [
        {"$match": {"document_id": {"$type": "string"}, "filename": {"$type": "string"}}},
        {"$group": {"_id": "$document_id", "filename": {"$first": "$filename"},
                    "chunks": {"$sum": 1}}},
        {"$sort": {"filename": 1, "_id": 1}},
    ]
    return [{"document_id": r["_id"], "filename": r["filename"], "chunks": r["chunks"]}
            for r in collection.aggregate(pipeline)]


def ingest_pdf(collection, filename: str, data: bytes) -> dict:
    # Always scope reads and writes to a content-derived stable document ID.
    filename = Path(filename).name
    if not filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported in this version")
    if not data or len(data) > MAX_BYTES:
        raise ValueError("PDF must be non-empty and 8 MB or smaller")
    doc_id = "pdf_" + hashlib.sha256(data).hexdigest()[:24]
    if collection.count_documents({"document_id": doc_id}, limit=1):
        return {"document_id": doc_id, "filename": filename, "status": "already indexed"}
    if len(list_documents(collection)) >= MAX_DOCS:
        raise ValueError(f"This demo supports at most {MAX_DOCS} uploaded documents")
    pages = extract_pages(data)
    chunks = make_chunks(pages)
    vectors = embed_passages(chunks)  # Generate all vectors before any DB writes.
    operations = []
    for chunk, vector in zip(chunks, vectors):
        record = {**chunk, "document_id": doc_id, "filename": filename,
                  "embedding_model": "jina-embeddings-v3", "embedding_jina": vector}
        operations.append(ReplaceOne({"document_id": doc_id, "chunk_id": chunk["chunk_id"]},
                                     record, upsert=True))
    if operations:
        collection.bulk_write(operations, ordered=True)
    return {"document_id": doc_id, "filename": filename, "status": "indexed", "chunks": len(chunks)}


def retrieve_for_document(collection, question: str, document_id: str, k: int = 8) -> list[dict]:
    if not document_id:
        raise ValueError("Select a document first")
    query_vector = embed_question(question)
    pipeline = [
        {"$vectorSearch": {"index": INDEX_NAME, "path": "embedding_jina", "queryVector": query_vector,
                           "numCandidates": max(40, k * 20), "limit": k,
                           "filter": {"document_id": document_id}}},
        {"$project": {"_id": 0, "document_id": 1, "filename": 1,
                      "page": 1, "chunk_id": 1, "text": 1,
                      "score": {"$meta": "vectorSearchScore"}}},
    ]
    return list(collection.aggregate(pipeline))


def answer_question(collection, client: Groq, document_id: str, question: str):
    chunks = retrieve_for_document(collection, question, document_id)
    draft = generate_answer(question, chunks, client)
    answer = validate_answer(question, draft, chunks, client)
    return answer, chunks
