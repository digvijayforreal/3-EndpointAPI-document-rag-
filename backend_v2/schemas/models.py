
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    document_id: str
    question: str = Field(min_length=1, max_length=2000)


class Source(BaseModel):
    source_id: int
    filename: str
    page: int | None = None
    chunk_id: int
    score: float
    text: str


class ChatResponse(BaseModel):
    answer: str
    document_id: str
    sources: list[Source]
