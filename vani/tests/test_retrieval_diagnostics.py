import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from fastapi import HTTPException

from api import main, vector_store


class FakeModel:
    def embed(self, _texts):
        yield SimpleNamespace(tolist=lambda: [0.1, 0.2])


class FakeQdrant:
    def __init__(self):
        self.query = None
        self.scroll_call = None

    def collection_exists(self, _name):
        return True

    def query_points(self, **kwargs):
        self.query = kwargs
        return SimpleNamespace(points=[SimpleNamespace(
            payload={"document_id": DOC_ID, "page_number": 1, "chunk_index": 2, "text": "author evidence"},
            score=0.8,
        )])

    def scroll(self, **kwargs):
        self.scroll_call = kwargs
        return ([SimpleNamespace(payload={
            "document_id": DOC_ID, "page_number": 1, "chunk_index": 0, "text": "Paper title. Author: A. Writer. Abstract: topic."
        })], None)


DOC_ID = str(uuid4())


class RetrievalDiagnosticTests(unittest.TestCase):
    def test_search_and_page_one_reads_are_filtered_to_exact_document(self):
        fake = FakeQdrant()
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(vector_store, "DATA_PATH", Path(directory)), \
                patch.object(vector_store, "_open_qdrant_client", return_value=fake), \
                patch.object(vector_store, "get_embedding_model", return_value=FakeModel()):
            matches = vector_store.diagnostic_search_chunks("question", 10, DOC_ID)
            first_page = vector_store.diagnostic_first_page_chunks(DOC_ID)

        must = fake.query["query_filter"].must
        self.assertEqual(len(must), 1)
        self.assertEqual((must[0].key, must[0].match.value), ("document_id", DOC_ID))
        page_must = fake.scroll_call["scroll_filter"].must
        self.assertEqual([(item.key, item.match.value) for item in page_must], [
            ("document_id", DOC_ID), ("page_number", 1),
        ])
        self.assertEqual(fake.query["limit"], 10)
        self.assertEqual(matches[0]["score"], 0.8)
        self.assertEqual(first_page[0]["text"], "Paper title. Author: A. Writer. Abstract: topic.")

    def test_route_disabled_by_default_and_does_not_retrieve(self):
        request = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))
        with patch.dict("os.environ", {}, clear=True), \
                patch.object(main, "diagnostic_first_page_chunks") as first_page:
            with self.assertRaises(HTTPException) as raised:
                main.retrieval_diagnostic(DOC_ID, request)
        self.assertEqual(raised.exception.status_code, 404)
        first_page.assert_not_called()

    def test_response_is_limited_to_safe_diagnostic_fields(self):
        request = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))
        first = {"document_id": DOC_ID, "page_number": 1, "chunk_index": 0, "text": "x" * 400}
        match = {"document_id": DOC_ID, "page_number": 2, "chunk_index": 4, "score": 0.7, "text": "short excerpt"}
        with patch.dict("os.environ", {"VANI_ENABLE_RETRIEVAL_DIAGNOSTICS": "1"}), \
                patch.object(main, "diagnostic_first_page_chunks", return_value=[first]), \
                patch.object(main, "diagnostic_search_chunks", return_value=[match]) as search:
            result = main.retrieval_diagnostic(DOC_ID, request)

        self.assertEqual(search.call_count, 3)
        self.assertTrue(all(call.args[2] == DOC_ID and call.args[1] == 10 for call in search.call_args_list))
        self.assertEqual(len(result.questions), 3)
        self.assertEqual(result.questions[0].question, "ใครเขียนเปเปอร์นี้")
        self.assertEqual(result.first_page_chunks[0].excerpt, "x" * main.DIAGNOSTIC_EXCERPT_CHARS)
        self.assertEqual(result.questions[0].matches[0].score, 0.7)
        self.assertEqual(set(result.questions[0].matches[0].model_dump()), {
            "document_id", "page_number", "chunk_index", "score", "excerpt",
        })
        self.assertFalse(result.first_page_truncated)

    def test_route_requires_canonical_uuid(self):
        request = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))
        with patch.dict("os.environ", {"VANI_ENABLE_RETRIEVAL_DIAGNOSTICS": "1"}):
            with self.assertRaises(HTTPException) as raised:
                main.retrieval_diagnostic(DOC_ID.upper(), request)
        self.assertEqual(raised.exception.status_code, 422)


if __name__ == "__main__":
    unittest.main()
