"""Request and response models for the Vanilla RAG API."""

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    page_count: int
    chunk_count: int
    duplicate: bool = False


class IndexedDocument(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    pdf_available: bool


class DocumentListResponse(BaseModel):
    documents: list[IndexedDocument]


class DocumentPdfResponse(BaseModel):
    document_id: str
    pdf_available: bool


class DocumentDeleteResponse(BaseModel):
    document_id: str
    deleted: bool


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_id: str | None = None


class Citation(BaseModel):
    source_id: str
    document_id: str
    filename: str
    page_number: int
    excerpt: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]


class SearchRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)
    document_id: str | None = None


class SearchResult(BaseModel):
    text: str
    score: float
    document_id: str
    filename: str
    page_number: int
    chunk_index: int


class SearchResponse(BaseModel):
    results: list[SearchResult]


class RetrievalDiagnosticMatch(BaseModel):
    document_id: str
    page_number: int
    chunk_index: int
    score: float | None = None
    excerpt: str


class RetrievalDiagnosticQuestion(BaseModel):
    question: str
    matches: list[RetrievalDiagnosticMatch]


class RetrievalDiagnosticResponse(BaseModel):
    document_id: str
    first_page_chunks: list[RetrievalDiagnosticMatch]
    first_page_chunk_count: int
    first_page_truncated: bool
    questions: list[RetrievalDiagnosticQuestion]
    interpretation: str
