
from fastapi import FastAPI, HTTPException
from pymongo.errors import PyMongoError

from backend_v2.database.mongodb import check_connection
from backend_v2.api.upload import router as upload_router

from backend_v2.api.embedding import router as embedding_router

from backend_v2.api.chat import router as chat_router

from pathlib import Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from backend_v2.api.documents import router as documents_router

app = FastAPI(
    title="Document RAG API V2",
    description="FastAPI backend for PDF/DOCX RAG",
    version="2.0.0"
)

# Register the Upload API
app.include_router(upload_router)
app.include_router(embedding_router)
app.include_router(upload_router)
app.include_router(embedding_router)
app.include_router(chat_router)
app.include_router(documents_router)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app.mount(
    "/static",
    StaticFiles(directory=str(FRONTEND_DIR)),
    name="static"
)

@app.get("/app", include_in_schema=False)
def frontend():
    return FileResponse(FRONTEND_DIR / "index.html")

@app.get("/")
def home():
    return {
        "message": "Document RAG API V2 is running"
    }


@app.get("/health")
def health_check():
    try:
        check_connection()

        return {
            "status": "connected",
            "database": "rag_database"
        }

    except PyMongoError:
        raise HTTPException(
            status_code=503,
            detail="MongoDB connection unavailable"
        )
