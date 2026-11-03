
import os
from urllib.parse import quote_plus

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

username = os.getenv("MONGODB_USERNAME")
password = os.getenv("MONGODB_PASSWORD")
host = os.getenv("MONGODB_HOST")

if not all([username, password, host]):
    raise ValueError("Missing MongoDB credentials in .env")

# Encode special characters safely
username = quote_plus(username)
password = quote_plus(password)

uri = f"mongodb+srv://{username}:{password}@{host}/?appName=Cluster0"

client = MongoClient(
    uri,
    serverSelectionTimeoutMS=10000
)

try:
    client.admin.command("ping")

    print("MongoDB Atlas connected successfully!")

    db = client["rag_database"]
    collection = db["documents"]

    print("Database:", db.name)
    print("Collection:", collection.name)

finally:
    client.close()
