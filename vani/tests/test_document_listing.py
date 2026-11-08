import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from api import vector_store


class FakeQdrant:
    def __init__(self, pages, has_collection=True):
        self.pages = pages
        self.has_collection = has_collection
        self.calls = []

    def collection_exists(self, name):
        return self.has_collection

    def scroll(self, **kwargs):
        self.calls.append(kwargs)
        return self.pages.pop(0)


class ListIndexedDocumentsTests(unittest.TestCase):
    def test_groups_documents_across_scroll_pages_without_text_payload(self):
        fake = FakeQdrant([
            ([
                SimpleNamespace(payload={"document_id": "doc-a", "filename": "paper.pdf"}),
                SimpleNamespace(payload={"document_id": "doc-b", "filename": "notes.pdf"}),
            ], "next-page"),
            ([
                SimpleNamespace(payload={"document_id": "doc-a", "filename": "paper.pdf"}),
                SimpleNamespace(payload={"document_id": "", "filename": "ignored.pdf"}),
            ], None),
        ])
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(vector_store, "DATA_PATH", Path(directory)), \
                patch.object(vector_store, "_open_qdrant_client", return_value=fake):
            documents = vector_store.list_indexed_documents()

        self.assertEqual(documents, [
            {"document_id": "doc-b", "filename": "notes.pdf", "chunk_count": 1},
            {"document_id": "doc-a", "filename": "paper.pdf", "chunk_count": 2},
        ])
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(fake.calls[0]["with_payload"], ["document_id", "filename"])
        self.assertFalse(fake.calls[0]["with_vectors"])
        self.assertEqual(fake.calls[1]["offset"], "next-page")

    def test_missing_collection_returns_empty_without_scroll(self):
        fake = FakeQdrant([], has_collection=False)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(vector_store, "DATA_PATH", Path(directory)), \
                patch.object(vector_store, "_open_qdrant_client", return_value=fake):
            self.assertEqual(vector_store.list_indexed_documents(), [])
        self.assertEqual(fake.calls, [])


if __name__ == "__main__":
    unittest.main()
