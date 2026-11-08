"""Safe, persistent storage for original PDF uploads."""

import os
import re
import tempfile
from hashlib import sha256
from pathlib import Path
from uuid import UUID

PDF_STORAGE_DIR = Path(__file__).resolve().parent.parent / "data" / "pdfs"
MAX_PDF_BYTES = 10 * 1024 * 1024
PDF_SIGNATURE = b"%PDF-"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class PdfStorageError(ValueError):
    """A validation or storage error safe to translate to an API response."""


class PdfAlreadyExists(PdfStorageError):
    """A PDF already exists for the requested document ID."""


def canonical_document_id(document_id: str) -> str:
    try:
        return str(UUID(document_id))
    except (ValueError, AttributeError, TypeError) as exc:
        raise PdfStorageError("Invalid document ID") from exc


def pdf_path(document_id: str) -> Path:
    canonical_id = canonical_document_id(document_id)
    root = PDF_STORAGE_DIR.resolve()
    path = root / f"{canonical_id}.pdf"
    if path.parent != root:
        raise PdfStorageError("Invalid PDF storage path")
    return path


def pdf_sha256_path(document_id: str) -> Path:
    canonical_id = canonical_document_id(document_id)
    root = PDF_STORAGE_DIR.resolve()
    path = root / f"{canonical_id}.sha256"
    if path.parent != root:
        raise PdfStorageError("Invalid PDF hash storage path")
    return path


def calculate_pdf_sha256(pdf_bytes: bytes) -> str:
    return sha256(pdf_bytes).hexdigest()


def store_pdf_sha256(document_id: str, digest: str) -> Path:
    """Atomically persist a validated SHA-256 sidecar for one canonical document."""
    if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
        raise PdfStorageError("Invalid PDF SHA-256 digest")
    path = pdf_sha256_path(document_id)
    PDF_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{canonical_document_id(document_id)}.",
        suffix=".sha256.tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as temporary_file:
            temporary_file.write(digest + "\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_name, path)
        return path
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def get_or_create_pdf_sha256(document_id: str) -> str:
    """Hash the stored original bytes and ensure their sidecar matches them."""
    path = pdf_path(document_id)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError("Stored PDF is unavailable")
    digest = calculate_pdf_sha256(path.read_bytes())
    sidecar = pdf_sha256_path(document_id)
    existing = None
    if sidecar.is_file() and not sidecar.is_symlink():
        try:
            existing = sidecar.read_text(encoding="ascii").strip()
        except (OSError, UnicodeError):
            existing = None
    if existing != digest:
        store_pdf_sha256(document_id, digest)
    return digest


def pdf_sha256_exists(document_id: str) -> bool:
    path = pdf_sha256_path(document_id)
    return path.exists() or path.is_symlink()


def remove_pdf_sha256(document_id: str) -> None:
    """Remove only the canonical document's hash sidecar, if present."""
    path = pdf_sha256_path(document_id)
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def validate_pdf_bytes(pdf_bytes: bytes) -> None:
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise PdfStorageError("PDF exceeds the 10 MB size limit")
    if not pdf_bytes.startswith(PDF_SIGNATURE):
        raise PdfStorageError("File does not have a valid PDF signature")


def store_pdf(document_id: str, pdf_bytes: bytes) -> Path:
    """Atomically store a validated PDF without replacing an existing document file."""
    canonical_id = canonical_document_id(document_id)
    validate_pdf_bytes(pdf_bytes)
    PDF_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    destination = pdf_path(canonical_id)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{canonical_id}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as temporary_file:
            temporary_file.write(pdf_bytes)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        try:
            # A same-directory hard link publishes the complete file atomically and
            # fails if another request has already stored this document ID.
            os.link(temporary_name, destination)
        except FileExistsError as exc:
            raise PdfAlreadyExists("A PDF is already stored for this document") from exc
        return destination
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def pdf_exists(document_id: str) -> bool:
    path = pdf_path(document_id)
    return path.is_file() and not path.is_symlink()


def remove_pdf(document_id: str) -> None:
    """Remove the just-created file when a new upload fails before API success."""
    path = pdf_path(document_id)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
