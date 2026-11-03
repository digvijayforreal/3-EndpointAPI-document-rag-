
import os
import requests
from urllib.parse import quote_plus

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

# 1. Get the user's question
question = input("Ask a question about the PDF: ").strip()

if not question:
    raise ValueError("Please enter a question.")

# 2. Generate the question embedding
print("\nGenerating question embedding...")

response = requests.post(
    "https://api.jina.ai/v1/embeddings",
    headers={
        "Authorization": f"Bearer {os.environ['JINA_API_KEY']}",
        "Content-Type": "application/json"
    },
    json={
        "model": "jina-embeddings-v3",
        "task": "retrieval.query",
        "dimensions": 1024,
        "input": [question]
    },
    timeout=60
)

response.raise_for_status()

query_vector = response.json()["data"][0]["embedding"]

if len(query_vector) != 1024:
    raise ValueError("Invalid embedding dimensions")

print("Question embedding generated!")

# 3. Connect to MongoDB Atlas
username = quote_plus(os.environ["MONGODB_USERNAME"])
password = quote_plus(os.environ["MONGODB_PASSWORD"])
host = os.environ["MONGODB_HOST"]

uri = (
    f"mongodb+srv://{username}:{password}"
    f"@{host}/?appName=Cluster0"
)

client = MongoClient(
    uri,
    serverSelectionTimeoutMS=10000
)

try:
    client.admin.command("ping")

    db = client[os.getenv("DB_NAME", "rag_database")]
    collection = db[
        os.getenv("COLLECTION_NAME", "documents")
    ]

    # 4. Search for matching document chunks
    pipeline = [
        {
            "$vectorSearch": {
                "index": "jina_vector_index",
                "path": "embedding_jina",
                "queryVector": query_vector,
                "numCandidates": 80,
                "limit": 3
            }
        },
        {
            "$project": {
                "_id": 0,
                "document_id": 1,
                "filename": 1,
                "page": 1,
                "chunk_id": 1,
                "text": 1,
                "score": {
                    "$meta": "vectorSearchScore"
                }
            }
        }
    ]

    results = list(collection.aggregate(pipeline))

    # 5. Display retrieved evidence
    print("\n--- RETRIEVAL RESULTS ---")

    if not results:
        print("No matching chunks found.")

    for i, result in enumerate(results, start=1):
        print(f"\nResult {i}")
        print("File:", result["filename"])
        print("Page:", result["page"])
        print("Chunk:", result["chunk_id"])
        print("Score:", round(result["score"], 4))
        print("Text:", result["text"])

finally:
    client.close()
