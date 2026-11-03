
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pymongo.errors import PyMongoError

from backend_v2.database.mongodb import (
    documents_collection,
    chunks_collection
)
from backend_v2.services.parser import extract_text
from backend_v2.services.chunker import create_chunks


router = APIRouter(tags=["Documents"])

MAX_FILE_SIZE = 8 * 1024 * 1024


@router.post("/upload", status_code=201)
async def upload_document(file: UploadFile = File(...)):

    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()

    if extension not in [".pdf", ".docx"]:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and DOCX files are supported"
        )

    data = await file.read(MAX_FILE_SIZE + 1)
    await file.close()

    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File size must be 8 MB or smaller"
        )

    if not data:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty"
        )

    try:
        sections = extract_text(data, extension)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Unable to extract text from this document"
        )

    chunks = create_chunks(sections)

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No readable text found in document"
        )

    document_id = str(uuid.uuid4())

    chunk_records = []

    for chunk in chunks:
        chunk_records.append({
            "document_id": document_id,
            "filename": filename,
            "chunk_id": chunk["chunk_id"],
            "page": chunk["page"],
            "text": chunk["text"],
            "embedding_status": "pending"
        })

    try:
        # Store extracted chunks
        chunks_collection.insert_many(chunk_records)

        # Store document information
        documents_collection.insert_one({
            "document_id": document_id,
            "filename": filename,
            "file_type": extension,
            "total_chunks": len(chunks),
            "embedding_status": "pending"
        })

    except PyMongoError:
        # Remove records if the upload was only partly saved
        try:
            chunks_collection.delete_many({
                "document_id": document_id
            })
            documents_collection.delete_many({
                "document_id": document_id
            })
        except PyMongoError:
            pass

        raise HTTPException(
            status_code=500,
            detail="Failed to save document"
        )

    return {
        "message": "Document uploaded successfully",
        "document_id": document_id,
        "filename": filename,
        "total_chunks": len(chunks),
        "embedding_status": "pending"
    }
