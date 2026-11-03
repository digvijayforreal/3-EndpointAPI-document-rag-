
import os
import requests
from urllib.parse import quote_plus

from dotenv import load_dotenv
from pymongo import MongoClient

from ingestion.loader import load_pdf
from ingestion.chunker import chunk_text

load_dotenv()

# -----------------------------------
# 1. Load and chunk the PDF
# -----------------------------------

print("Loading PDF...")

pages = load_pdf("documents/sample.pdf")
chunks = chunk_text(pages)

if not chunks:
    raise ValueError("No text chunks found in the PDF")

print(f"Generated {len(chunks)} chunks")

# -----------------------------------
# 2. Generate embeddings using Jina
# -----------------------------------

print("Generating cloud embeddings...")

api_key = os.environ["JINA_API_KEY"]

response = requests.post(
    "https://api.jina.ai/v1/embeddings",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    },
    json={
        "model": "jina-embeddings-v3",
        "task": "retrieval.passage",
        "dimensions": 1024,
        "input": [chunk["text"] for chunk in chunks]
    },
    timeout=120
)

response.raise_for_status()
result = response.json()

# Preserve the original input order
embeddings = sorted(
    result["data"],
    key=lambda item: item["index"]
)

if len(embeddings) != len(chunks):
    raise ValueError("Embedding count does not match chunk count")

# -----------------------------------
# 3. Connect to MongoDB Atlas
# -----------------------------------

username = quote_plus(os.environ["MONGODB_USERNAME"])
password = quote_plus(os.environ["MONGODB_PASSWORD"])
host = os.environ["MONGODB_HOST"]

uri = f"mongodb+srv://{username}:{password}@{host}/?appName=Cluster0"

client = MongoClient(uri, serverSelectionTimeoutMS=10000)

try:
    client.admin.command("ping")

    db = client[os.getenv("DB_NAME", "rag_database")]
    collection = db[
        os.getenv("COLLECTION_NAME", "documents")
    ]

    # -----------------------------------
    # 4. Store chunks and embeddings
    # -----------------------------------

    for chunk, item in zip(chunks, embeddings):
        vector = item["embedding"]

        if len(vector) != 1024:
            raise ValueError("Expected a 1024-dimensional vector")

        document = {
            "document_id": "sample_pdf",
            "filename": "sample.pdf",
            "chunk_id": chunk["chunk_id"],
            "page": chunk["page"],
            "text": chunk["text"],
            "embedding_jina": vector,
            "embedding_model": "jina-embeddings-v3"
        }

        collection.replace_one(
            {
                "document_id": "sample_pdf",
                "chunk_id": chunk["chunk_id"]
            },
            document,
            upsert=True
        )

        print(f"Stored chunk {chunk['chunk_id']}")

    print("\nSUCCESS!")
    print("Stored chunks:", len(chunks))
    print("Vector dimensions:", 1024)
    print("Database:", db.name)
    print("Collection:", collection.name)

finally:
    client.close()
