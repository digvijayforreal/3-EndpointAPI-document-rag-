
import logging

from fastapi import APIRouter, HTTPException
from backend_v2.schemas.models import ChatRequest, ChatResponse
from backend_v2.services.rag_service import ask_document

router = APIRouter(tags=["Chat"])
logger = logging.getLogger(__name__)


@router.post("/ask", response_model=ChatResponse)
def ask_question(request: ChatRequest):
    try:
        return ask_document(
            document_id=request.document_id,
            question=request.question
        )

    except ValueError as error:
        message = str(error)

        status_code = (
            404 if message == "Document not found" else 400
        )

        raise HTTPException(
            status_code=status_code,
            detail=message
        )

    except Exception:
        logger.exception("RAG request failed")

        raise HTTPException(
            status_code=502,
            detail="Unable to process the question"
        )
