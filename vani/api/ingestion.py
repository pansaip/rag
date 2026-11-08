"""Extract page text and create metadata-rich chunks from a PDF."""

from io import BytesIO
from typing import TypedDict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


class ChunkMetadata(TypedDict):
    document_id: str
    filename: str
    page_number: int
    chunk_index: int


class DocumentChunk(TypedDict):
    text: str
    metadata: ChunkMetadata


class PdfIngestionError(ValueError):
    """Raised when a PDF cannot be read or contains no extractable text."""


def extract_and_chunk_pdf(
    pdf_bytes: bytes, document_id: str, filename: str
) -> tuple[int, list[DocumentChunk]]:
    """Extract text per page and split it while retaining source metadata."""
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        if reader.is_encrypted:
            raise PdfIngestionError("Encrypted PDFs are not supported")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks: list[DocumentChunk] = []

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if not page_text:
                continue

            for chunk_text in splitter.split_text(page_text):
                chunks.append(
                    {
                        "text": chunk_text,
                        "metadata": {
                            "document_id": document_id,
                            "filename": filename,
                            "page_number": page_number,
                            "chunk_index": len(chunks),
                        },
                    }
                )

        if not chunks:
            raise PdfIngestionError("PDF contains no extractable text")
        return len(reader.pages), chunks
    except PdfIngestionError:
        raise
    except Exception as exc:
        raise PdfIngestionError("PDF is corrupted or cannot be processed") from exc
