import unittest
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from fastapi import HTTPException
from qdrant_client import models

from api import main, vector_store


class FakeQdrantClient:
    def __init__(self, collection_exists=True, result_status=models.UpdateStatus.COMPLETED):
        self.has_collection = collection_exists
        self.result_status = result_status
        self.delete_kwargs = None

    def collection_exists(self, collection_name):
        self.collection_name = collection_name
        return self.has_collection

    def delete(self, **kwargs):
        self.delete_kwargs = kwargs
        return SimpleNamespace(status=self.result_status)


class VectorStoreDeletionTests(unittest.TestCase):
    def test_deletes_using_exact_document_id_filter_and_existing_collection(self):
        document_id = str(uuid4())
        client = FakeQdrantClient()
        with patch.object(vector_store, "DATA_PATH") as data_path, \
                patch.object(vector_store, "_open_qdrant_client", return_value=client):
            data_path.exists.return_value = True
            self.assertTrue(vector_store.delete_indexed_document(document_id))

        kwargs = client.delete_kwargs
        self.assertEqual(kwargs["collection_name"], vector_store.COLLECTION_NAME)
        self.assertTrue(kwargs["wait"])
        selector = kwargs["points_selector"]
        self.assertIsInstance(selector, models.FilterSelector)
        self.assertEqual(len(selector.filter.must), 1)
        condition = selector.filter.must[0]
        self.assertEqual(condition.key, "document_id")
        self.assertEqual(condition.match.value, document_id)

    def test_missing_collection_never_calls_delete(self):
        client = FakeQdrantClient(collection_exists=False)
        with patch.object(vector_store, "DATA_PATH") as data_path, \
                patch.object(vector_store, "_open_qdrant_client", return_value=client):
            data_path.exists.return_value = True
            self.assertFalse(vector_store.delete_indexed_document(str(uuid4())))
        self.assertIsNone(client.delete_kwargs)


class DeleteDocumentApiTests(unittest.TestCase):
    def setUp(self):
        self.document_id = str(uuid4())

    def test_rejects_noncanonical_or_malformed_ids(self):
        with self.assertRaises(HTTPException) as raised:
            main.delete_document("../../bad")
        self.assertEqual(raised.exception.status_code, 422)

        with self.assertRaises(HTTPException) as raised:
            main.delete_document(self.document_id.upper())
        self.assertEqual(raised.exception.status_code, 422)

    def test_absent_document_returns_404_without_mutation(self):
        with patch.object(main, "indexed_document_exists", return_value=False), \
                patch.object(main, "pdf_exists", return_value=False), \
                patch.object(main, "pdf_sha256_exists", return_value=False), \
                patch.object(main, "delete_indexed_document") as delete_points, \
                patch.object(main, "remove_pdf") as remove_pdf:
            with self.assertRaises(HTTPException) as raised:
                main.delete_document(self.document_id)
        self.assertEqual(raised.exception.status_code, 404)
        delete_points.assert_not_called()
        remove_pdf.assert_not_called()

    def test_qdrant_failure_retains_pdf_and_returns_retryable_error(self):
        with patch.object(main, "indexed_document_exists", return_value=True), \
                patch.object(main, "pdf_exists", return_value=True), \
                patch.object(main, "pdf_sha256_exists", return_value=True), \
                patch.object(main, "delete_indexed_document", side_effect=RuntimeError("mock")), \
                patch.object(main, "remove_pdf") as remove_pdf:
            with self.assertRaises(HTTPException) as raised:
                main.delete_document(self.document_id)
        self.assertEqual(raised.exception.status_code, 503)
        self.assertIn("PDF and hash metadata were retained", raised.exception.detail)
        remove_pdf.assert_not_called()

    def test_pdf_cleanup_failure_is_reported_and_retry_finishes_without_points(self):
        with patch.object(main, "indexed_document_exists", return_value=True), \
                patch.object(main, "pdf_exists", return_value=True), \
                patch.object(main, "pdf_sha256_exists", return_value=True), \
                patch.object(main, "delete_indexed_document", return_value=True) as delete_points, \
                patch.object(main, "remove_pdf", side_effect=OSError("mock")):
            with self.assertRaises(HTTPException) as raised:
                main.delete_document(self.document_id)
        self.assertEqual(raised.exception.status_code, 503)
        self.assertIn("Retry the same document ID", raised.exception.detail)
        delete_points.assert_called_once_with(self.document_id)

        with patch.object(main, "indexed_document_exists", return_value=False), \
                patch.object(main, "pdf_exists", return_value=True), \
                patch.object(main, "pdf_sha256_exists", return_value=True), \
                patch.object(main, "delete_indexed_document") as delete_points, \
                patch.object(main, "remove_pdf") as remove_pdf:
            response = main.delete_document(self.document_id)
        self.assertTrue(response.deleted)
        delete_points.assert_not_called()
        remove_pdf.assert_called_once_with(self.document_id)

    def test_success_deletes_points_then_only_matching_pdf(self):
        order = []
        with patch.object(main, "indexed_document_exists", return_value=True), \
                patch.object(main, "pdf_exists", return_value=True), \
                patch.object(main, "pdf_sha256_exists", return_value=True), \
                patch.object(main, "delete_indexed_document", side_effect=lambda value: order.append(("qdrant", value)) or True), \
                patch.object(main, "remove_pdf", side_effect=lambda value: order.append(("pdf", value))):
            response = main.delete_document(self.document_id)
        self.assertEqual(order, [("qdrant", self.document_id), ("pdf", self.document_id)])
        self.assertEqual(response.document_id, self.document_id)
        self.assertTrue(response.deleted)


if __name__ == "__main__":
    unittest.main()
