"""Persistent local Qdrant storage with local FastEmbed vectors."""

from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

from api.ingestion import DocumentChunk

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_SIZE = 384
COLLECTION_NAME = "vani_documents"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "qdrant"


@lru_cache(maxsize=1)
def get_embedding_model() -> TextEmbedding:
    """Load the supported model once, on the first embedding request."""
    supported = TextEmbedding.list_supported_models()
    model_info = next((item for item in supported if item["model"] == MODEL_NAME), None)
    if model_info is None or model_info["dim"] != VECTOR_SIZE:
        raise RuntimeError("Configured FastEmbed model is unavailable or has unexpected dimensions")
    return TextEmbedding(model_name=MODEL_NAME)


@lru_cache(maxsize=1)
def _open_qdrant_client() -> QdrantClient:
    """Open the local Qdrant database without changing its collections."""
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(DATA_PATH))


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Open persistent local storage and create the MVP collection once."""
    client = _open_qdrant_client()
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )
    return client


def list_indexed_documents() -> list[dict[str, Any]]:
    """List document metadata from existing chunks without creating collections or reading text."""
    if not DATA_PATH.exists():
        return []

    client = _open_qdrant_client()
    if not client.collection_exists(COLLECTION_NAME):
        return []

    documents: dict[str, dict[str, Any]] = {}
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            offset=offset,
            limit=256,
            with_payload=["document_id", "filename"],
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            document_id = payload.get("document_id")
            if not isinstance(document_id, str) or not document_id:
                continue

            filename = payload.get("filename")
            if not isinstance(filename, str) or not filename:
                filename = "Unknown filename"
            document = documents.setdefault(
                document_id,
                {"document_id": document_id, "filename": filename, "chunk_count": 0},
            )
            document["chunk_count"] += 1

        if next_offset is None:
            break
        offset = next_offset

    return sorted(documents.values(), key=lambda item: (item["filename"].casefold(), item["document_id"]))


def indexed_document_exists(document_id: str) -> bool:
    """Check for one exact document ID using Qdrant scroll without changing points."""
    if not DATA_PATH.exists():
        return False
    client = _open_qdrant_client()
    if not client.collection_exists(COLLECTION_NAME):
        return False

    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                )
            ]
        ),
        limit=1,
        with_payload=["document_id"],
        with_vectors=False,
    )
    return any((point.payload or {}).get("document_id") == document_id for point in points)


def delete_indexed_document(document_id: str) -> bool:
    """Delete points for exactly one document ID; return false if collection is absent."""
    if not DATA_PATH.exists():
        return False
    client = _open_qdrant_client()
    if not client.collection_exists(COLLECTION_NAME):
        return False

    document_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="document_id",
                match=models.MatchValue(value=document_id),
            )
        ]
    )
    result = client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(filter=document_filter),
        wait=True,
    )
    if getattr(result, "status", None) != models.UpdateStatus.COMPLETED:
        raise RuntimeError("Qdrant did not confirm document deletion")
    return True


def store_chunks(chunks: list[DocumentChunk]) -> None:
    """Embed chunks and store their text and citation metadata in Qdrant."""
    model = get_embedding_model()
    texts = [chunk["text"] for chunk in chunks]
    vectors = model.embed(texts)
    points = [
        models.PointStruct(
            id=str(uuid4()),
            vector=vector.tolist(),
            payload={"text": chunk["text"], **chunk["metadata"]},
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    get_qdrant_client().upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )


def search_chunks(
    question: str, top_k: int = 5, document_id: str | None = None
) -> list[dict[str, Any]]:
    """Return nearest chunks with their scores and citation metadata."""
    model = get_embedding_model()
    query_vector = next(model.embed([question])).tolist()

    query_filter = None
    if document_id is not None:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                )
            ]
        )

    response = get_qdrant_client().query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )
    results: list[dict[str, Any]] = []
    for point in response.points:
        payload = point.payload or {}
        results.append(
            {
                "text": payload.get("text", ""),
                "score": point.score,
                "document_id": payload.get("document_id", ""),
                "filename": payload.get("filename", ""),
                "page_number": payload.get("page_number", 0),
                "chunk_index": payload.get("chunk_index", 0),
            }
        )
    return results


def diagnostic_search_chunks(question: str, top_k: int, document_id: str) -> list[dict[str, Any]]:
    """Search one document without creating a collection or opening another client."""
    if not DATA_PATH.exists():
        return []
    client = _open_qdrant_client()
    if not client.collection_exists(COLLECTION_NAME):
        return []

    model = get_embedding_model()
    query_vector = next(model.embed([question])).tolist()
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=models.Filter(
            must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
        ),
        limit=top_k,
        with_payload=True,
    )
    return [
        {
            "document_id": (point.payload or {}).get("document_id", ""),
            "page_number": (point.payload or {}).get("page_number", 0),
            "chunk_index": (point.payload or {}).get("chunk_index", 0),
            "score": point.score,
            "text": (point.payload or {}).get("text", ""),
        }
        for point in response.points
    ]


def diagnostic_first_page_chunks(document_id: str) -> list[dict[str, Any]]:
    """Read all indexed chunks from page one of exactly one document."""
    if not DATA_PATH.exists():
        return []
    client = _open_qdrant_client()
    if not client.collection_exists(COLLECTION_NAME):
        return []

    document_filter = models.Filter(
        must=[
            models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)),
            models.FieldCondition(key="page_number", match=models.MatchValue(value=1)),
        ]
    )
    chunks: list[dict[str, Any]] = []
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=document_filter,
            offset=offset,
            limit=100,
            with_payload=["document_id", "filename", "page_number", "chunk_index", "text"],
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            if payload.get("document_id") != document_id or payload.get("page_number") != 1:
                continue
            chunks.append(
                {
                    "document_id": document_id,
                    "page_number": 1,
                    "chunk_index": payload.get("chunk_index", 0),
                    "filename": payload.get("filename", ""),
                    "text": payload.get("text", ""),
                }
            )
        if next_offset is None:
            break
        offset = next_offset
    return sorted(chunks, key=lambda chunk: chunk["chunk_index"])
