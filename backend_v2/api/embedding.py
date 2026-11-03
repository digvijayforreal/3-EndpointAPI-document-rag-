
from fastapi import APIRouter, HTTPException
from pymongo.errors import PyMongoError
from requests.exceptions import RequestException

from backend_v2.services.embedding_service import embed_document

router = APIRouter(tags=["Embeddings"])


@router.post("/embed/{document_id}")
def create_document_embeddings(document_id: str):
    try:
        return embed_document(document_id)

    except ValueError as error:
        raise HTTPException(
            status_code=404 if str(error) == "Document not found" else 502,
            detail=str(error)
        )

    except (RequestException, PyMongoError):
        raise HTTPException(
            status_code=502,
            detail="Embedding service or database error"
        )

    except RuntimeError:
        raise HTTPException(
            status_code=500,
            detail="Embedding service configuration error"
        )
