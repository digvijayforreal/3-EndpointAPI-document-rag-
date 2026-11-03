
import os
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables
BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

# Get MongoDB credentials
username = os.getenv("MONGODB_USERNAME")
password = os.getenv("MONGODB_PASSWORD")
host = os.getenv("MONGODB_HOST")

db_name = os.getenv("DB_NAME", "rag_database")

if not all([username, password, host]):
    raise RuntimeError("MongoDB credentials missing in .env")

# Encode username and password safely
username = quote_plus(username)
password = quote_plus(password)

# Build MongoDB Atlas connection URI
MONGO_URI = (
    f"mongodb+srv://{username}:{password}@{host}/"
    "?retryWrites=true&w=majority"
)

# Connect to MongoDB
client = MongoClient(
    MONGO_URI,
    serverSelectionTimeoutMS=5000
)

db = client[db_name]

# Separate V2 collections
documents_collection = db["documents_v2"]
chunks_collection = db["chunks_v2"]


def check_connection():
    client.admin.command("ping")
    return True
