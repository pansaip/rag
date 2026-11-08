"""FastAPI backend for the Vanilla RAG application."""

import logging
import os
import threading
from uuid import UUID, uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from api import pdf_storage
from api import observability
from api.ingestion import PdfIngestionError, extract_and_chunk_pdf
from api.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentListResponse,
    DocumentDeleteResponse,
    DocumentPdfResponse,
    DocumentUploadResponse,
    SearchRequest,
    SearchResponse,
    RetrievalDiagnosticMatch,
    RetrievalDiagnosticQuestion,
    RetrievalDiagnosticResponse,
)
from api.pdf_storage import (
    MAX_PDF_BYTES,
    PdfAlreadyExists,
    PdfStorageError,
    canonical_document_id,
    calculate_pdf_sha256,
    get_or_create_pdf_sha256,
    pdf_exists,
    pdf_sha256_exists,
    pdf_path,
    remove_pdf,
    remove_pdf_sha256,
    store_pdf_sha256,
    store_pdf,
)
from api.vector_store import (
    delete_indexed_document,
    diagnostic_first_page_chunks,
    diagnostic_search_chunks,
    indexed_document_exists,
    list_indexed_documents,
    search_chunks,
    store_chunks,
)
from api.qa import RAGServiceError, answer_question

app = FastAPI(title="Vanilla RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type"],
)
logger = logging.getLogger(__name__)
MAX_FILE_SIZE = MAX_PDF_BYTES
# The local deployment uses one API process. This lock serializes upload, attachment,
# and deletion operations across its request threads; it is not shared across workers.
_document_mutation_lock = threading.RLock()


def _find_document_with_pdf_sha256(digest: str) -> dict | None:
    """Find an indexed document by hashing only its saved original PDF."""
    documents = list_indexed_documents()
    indexed_ids = {document["document_id"] for document in documents}

    if pdf_storage.PDF_STORAGE_DIR.exists():
        for sidecar in pdf_storage.PDF_STORAGE_DIR.glob("*.sha256"):
            document_id = sidecar.name.removesuffix(".sha256")
            try:
                if canonical_document_id(document_id) != document_id:
                    continue
            except PdfStorageError:
                continue
            if document_id not in indexed_ids or not pdf_exists(document_id):
                remove_pdf_sha256(document_id)

    for document in documents:
        document_id = document["document_id"]
        try:
            canonical_id = canonical_document_id(document_id)
        except PdfStorageError:
            continue
        if canonical_id != document_id or not pdf_exists(document_id):
            continue
        if get_or_create_pdf_sha256(document_id) == digest:
            return document
    return None


async def _read_pdf_upload(file: UploadFile) -> bytes:
    contents = bytearray()
    while True:
        remaining = MAX_FILE_SIZE + 1 - len(contents)
        if remaining <= 0:
            raise HTTPException(status_code=413, detail="PDF must be 10 MB or smaller")
        part = await file.read(min(64 * 1024, remaining))
        if not part:
            break
        contents.extend(part)

    pdf_bytes = bytes(contents)
    if not pdf_bytes.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="File does not have a valid PDF signature")
    return pdf_bytes


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the API process is running."""
    return {"status": "ok"}


@app.get("/documents", response_model=DocumentListResponse)
def list_documents() -> DocumentListResponse:
    """List indexed document metadata without returning stored passage text."""
    try:
        documents = list_indexed_documents()
        for document in documents:
            document["pdf_available"] = pdf_exists(document["document_id"])
        return DocumentListResponse(documents=documents)
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Document listing is temporarily unavailable",
        ) from None


@app.delete("/documents/{document_id}", response_model=DocumentDeleteResponse)
def delete_document(document_id: str) -> DocumentDeleteResponse:
    """Delete one exact indexed document and its matching saved PDF."""
    try:
        canonical_id = canonical_document_id(document_id)
    except PdfStorageError:
        raise HTTPException(status_code=422, detail="document_id must be a valid canonical UUID") from None
    if canonical_id != document_id:
        raise HTTPException(status_code=422, detail="document_id must be a valid canonical UUID")

    with _document_mutation_lock:
        try:
            indexed = indexed_document_exists(canonical_id)
            stored_pdf = pdf_exists(canonical_id)
            stored_hash = pdf_sha256_exists(canonical_id)
        except Exception:
            raise HTTPException(status_code=503, detail="Document lookup failed; no deletion was confirmed. Retry the same document ID.") from None

        if not indexed and not stored_pdf and not stored_hash:
            raise HTTPException(status_code=404, detail="Document not found")

        if indexed:
            try:
                collection_available = delete_indexed_document(canonical_id)
                if not collection_available:
                    raise RuntimeError("Qdrant collection is unavailable")
            except Exception:
                raise HTTPException(status_code=503, detail="Indexed document deletion failed; the PDF and hash metadata were retained. Retry the same document ID.") from None

        try:
            remove_pdf(canonical_id)
        except Exception:
            raise HTTPException(status_code=503, detail="Indexed points were removed, but PDF cleanup failed. Retry the same document ID to finish cleanup.") from None
        try:
            remove_pdf_sha256(canonical_id)
        except Exception:
            raise HTTPException(status_code=503, detail="PDF cleanup completed, but hash metadata cleanup failed. Retry the same document ID to finish cleanup.") from None

    return DocumentDeleteResponse(document_id=canonical_id, deleted=True)


@app.get("/documents/{document_id}/pdf")
def get_document_pdf(document_id: str) -> FileResponse:
    """Serve the original PDF for an existing indexed document."""
    try:
        canonical_id = canonical_document_id(document_id)
    except PdfStorageError:
        raise HTTPException(status_code=422, detail="document_id must be a valid UUID") from None

    try:
        if not indexed_document_exists(canonical_id):
            raise HTTPException(status_code=404, detail="Document not found")
        path = pdf_path(canonical_id)
        if not pdf_exists(canonical_id):
            raise HTTPException(status_code=404, detail="Original PDF is not available")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=503, detail="Document storage is temporarily unavailable") from None

    return FileResponse(
        path,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="{canonical_id}.pdf"',
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.post("/documents", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    """Index a PDF once, reusing an existing document for byte-identical uploads."""
    filename = file.filename or ""
    try:
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="File must have a .pdf extension")
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Content-Type must be application/pdf")

        pdf_bytes = await _read_pdf_upload(file)
        digest = calculate_pdf_sha256(pdf_bytes)

        with _document_mutation_lock:
            try:
                existing = _find_document_with_pdf_sha256(digest)
            except Exception:
                raise HTTPException(status_code=503, detail="Duplicate document lookup is temporarily unavailable; no indexing was started.") from None
            if existing:
                return DocumentUploadResponse(
                    document_id=existing["document_id"],
                    filename=existing["filename"],
                    status="reused",
                    page_count=0,
                    chunk_count=existing["chunk_count"],
                    duplicate=True,
                )

            document_id = str(uuid4())
            try:
                page_count, chunks = extract_and_chunk_pdf(pdf_bytes, document_id, filename)
            except PdfIngestionError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

            try:
                store_pdf(document_id, pdf_bytes)
                store_pdf_sha256(document_id, digest)
            except PdfAlreadyExists:
                raise HTTPException(status_code=409, detail="A PDF already exists for this document") from None
            except PdfStorageError:
                try:
                    remove_pdf(document_id)
                    remove_pdf_sha256(document_id)
                except OSError as cleanup_error:
                    logger.warning("PDF cleanup failed after metadata error type=%s", type(cleanup_error).__name__)
                raise HTTPException(status_code=503, detail="Original PDF or hash metadata storage is temporarily unavailable") from None
            except OSError:
                try:
                    remove_pdf(document_id)
                    remove_pdf_sha256(document_id)
                except OSError as cleanup_error:
                    logger.warning("PDF cleanup failed after metadata error type=%s", type(cleanup_error).__name__)
                raise HTTPException(status_code=503, detail="Original PDF or hash metadata storage is temporarily unavailable") from None

            try:
                store_chunks(chunks)
            except Exception as exc:
                try:
                    points_may_exist = indexed_document_exists(document_id)
                except Exception as lookup_error:
                    # Retain the original and digest if Qdrant's commit state is uncertain.
                    points_may_exist = True
                    logger.warning("Index state lookup failed after upload error type=%s", type(lookup_error).__name__)
                if not points_may_exist:
                    for cleanup in (remove_pdf_sha256, remove_pdf):
                        try:
                            cleanup(document_id)
                        except OSError as cleanup_error:
                            logger.warning("Upload cleanup failed after indexing error type=%s", type(cleanup_error).__name__)
                raise HTTPException(
                    status_code=503,
                    detail="Document indexing could not be confirmed. Retry the same PDF; duplicate checks will reuse it if any points were stored.",
                ) from exc

            return DocumentUploadResponse(
                document_id=document_id,
                filename=filename,
                status="stored",
                page_count=page_count,
                chunk_count=len(chunks),
                duplicate=False,
            )
    finally:
        await file.close()


@app.put("/documents/{document_id}/pdf", response_model=DocumentPdfResponse)
async def attach_document_pdf(document_id: str, file: UploadFile = File(...)) -> DocumentPdfResponse:
    """Persist a legacy document's original PDF without re-indexing it."""
    try:
        try:
            canonical_id = canonical_document_id(document_id)
        except PdfStorageError:
            raise HTTPException(status_code=422, detail="document_id must be a valid UUID") from None

        try:
            if not indexed_document_exists(canonical_id):
                raise HTTPException(status_code=404, detail="Document not found")
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=503, detail="Document lookup is temporarily unavailable") from None

        filename = file.filename or ""
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="File must have a .pdf extension")
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Content-Type must be application/pdf")
        pdf_bytes = await _read_pdf_upload(file)
        with _document_mutation_lock:
            try:
                if not indexed_document_exists(canonical_id):
                    raise HTTPException(status_code=404, detail="Document not found")
                store_pdf(canonical_id, pdf_bytes)
                store_pdf_sha256(canonical_id, calculate_pdf_sha256(pdf_bytes))
            except HTTPException:
                raise
            except PdfAlreadyExists:
                raise HTTPException(status_code=409, detail="An original PDF is already stored for this document") from None
            except (PdfStorageError, OSError):
                try:
                    remove_pdf(canonical_id)
                    remove_pdf_sha256(canonical_id)
                except OSError as cleanup_error:
                    logger.warning("Attachment cleanup failed type=%s", type(cleanup_error).__name__)
                raise HTTPException(status_code=503, detail="Original PDF or hash metadata storage is temporarily unavailable") from None
            except Exception:
                try:
                    remove_pdf(canonical_id)
                    remove_pdf_sha256(canonical_id)
                except OSError as cleanup_error:
                    logger.warning("Attachment cleanup failed type=%s", type(cleanup_error).__name__)
                raise HTTPException(status_code=503, detail="Document lookup or PDF storage is temporarily unavailable") from None
        return DocumentPdfResponse(document_id=canonical_id, pdf_available=True)
    finally:
        await file.close()


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Retrieve evidence and generate an answer grounded in those sources."""
    with observability.chat_request(os.getenv("GROQ_MODEL")):
        if not request.question.strip():
            raise HTTPException(status_code=422, detail="Question must not be blank")

        document_id = request.document_id
        if document_id is not None:
            try:
                document_id = str(UUID(document_id))
            except ValueError as exc:
                raise HTTPException(status_code=422, detail="document_id must be a valid UUID") from exc

        try:
            return ChatResponse(**answer_question(request.question, document_id))
        except RAGServiceError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.public_message) from None


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    """Find relevant stored chunks without generating an LLM answer."""
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="Question must not be blank")

    document_id = request.document_id
    if document_id is not None:
        try:
            document_id = str(UUID(document_id))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="document_id must be a valid UUID") from exc

    try:
        results = search_chunks(request.question, request.top_k, document_id)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Document search is temporarily unavailable",
        ) from exc
    return SearchResponse(results=results)


DIAGNOSTIC_QUESTIONS = ("ใครเขียนเปเปอร์นี้", "มันเกี่ยวกับอะไร", "deepsea คืออะไร")
DIAGNOSTIC_EXCERPT_CHARS = 280
DIAGNOSTIC_MAX_FIRST_PAGE_CHUNKS = 50


@app.get(
    "/diagnostics/retrieval/{document_id}",
    response_model=RetrievalDiagnosticResponse,
    include_in_schema=False,
)
def retrieval_diagnostic(document_id: str, request: Request) -> RetrievalDiagnosticResponse:
    """Local opt-in diagnostic; reads only chunks belonging to the requested document."""
    if os.getenv("VANI_ENABLE_RETRIEVAL_DIAGNOSTICS") != "1":
        raise HTTPException(status_code=404, detail="Not found")
    client_host = request.client.host if request.client else ""
    if client_host not in {"127.0.0.1", "::1", "localhost", "testclient"}:
        raise HTTPException(status_code=404, detail="Not found")
    try:
        canonical_id = str(UUID(document_id))
    except ValueError:
        raise HTTPException(status_code=422, detail="document_id must be a valid UUID") from None
    if canonical_id != document_id:
        raise HTTPException(status_code=422, detail="document_id must be a canonical UUID")

    def as_match(chunk: dict, include_score: bool) -> RetrievalDiagnosticMatch:
        return RetrievalDiagnosticMatch(
            document_id=canonical_id,
            page_number=int(chunk.get("page_number", 0)),
            chunk_index=int(chunk.get("chunk_index", 0)),
            score=chunk.get("score") if include_score else None,
            excerpt=str(chunk.get("text", "")).strip()[:DIAGNOSTIC_EXCERPT_CHARS],
        )

    try:
        first_page = diagnostic_first_page_chunks(canonical_id)
        questions = [
            RetrievalDiagnosticQuestion(
                question=question,
                matches=[as_match(chunk, True) for chunk in diagnostic_search_chunks(question, 10, canonical_id)],
            )
            for question in DIAGNOSTIC_QUESTIONS
        ]
    except Exception:
        raise HTTPException(status_code=503, detail="Retrieval diagnostic is temporarily unavailable") from None

    if not first_page:
        interpretation = (
            "No indexed page-one chunks were found for this document; investigate PDF extraction/indexing first. "
            "If page-one chunks exist but author/title/abstract evidence is absent from them, that is also an extraction/indexing issue."
        )
    else:
        interpretation = (
            "Inspect the page-one excerpts for title, author, and abstract text. If the needed evidence is present there "
            "but absent from a question's top matches, that indicates retrieval ranking; if absent from page one too, "
            "investigate extraction/indexing."
        )
    visible_first_page = first_page[:DIAGNOSTIC_MAX_FIRST_PAGE_CHUNKS]
    return RetrievalDiagnosticResponse(
        document_id=canonical_id,
        first_page_chunks=[as_match(chunk, False) for chunk in visible_first_page],
        first_page_chunk_count=len(first_page),
        first_page_truncated=len(first_page) > len(visible_first_page),
        questions=questions,
        interpretation=interpretation,
    )
