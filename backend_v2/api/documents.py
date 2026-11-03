
from fastapi import APIRouter
from backend_v2.database.mongodb import documents_collection

router = APIRouter(tags=["Documents"])


@router.get("/documents")
def get_documents():

    documents = documents_collection.find(
        {},
        {"_id": 0}
    )

    return {
        "documents": list(documents)
    }
