import asyncio
import concurrent.futures
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from fastapi import HTTPException

from api import main, pdf_storage


class FakeUpload:
    def __init__(self, content=b"%PDF-1.7 test", filename="test.pdf", content_type="application/pdf"):
        self.content = content
        self.filename = filename
        self.content_type = content_type
        self.closed = False

    async def read(self, size=-1):
        if not self.content:
            return b""
        chunk, self.content = self.content[:size], self.content[size:]
        return chunk

    async def close(self):
        self.closed = True


class PdfApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_patch = patch.object(pdf_storage, "PDF_STORAGE_DIR", Path(self.temp_dir.name) / "pdfs")
        self.storage_patch.start()

    def tearDown(self):
        self.storage_patch.stop()
        self.temp_dir.cleanup()

    def test_post_upload_persists_pdf_and_preserves_response(self):
        upload = FakeUpload()
        chunks = [{"text": "mock chunk", "metadata": {"document_id": "id", "filename": "test.pdf"}}]
        with patch.object(main, "extract_and_chunk_pdf", return_value=(3, chunks)), \
                patch.object(main, "store_chunks") as store_chunks, \
                patch.object(main, "_find_document_with_pdf_sha256", return_value=None):
            response = asyncio.run(main.upload_document(upload))

        self.assertEqual(response.status, "stored")
        self.assertEqual(response.page_count, 3)
        self.assertEqual(response.chunk_count, 1)
        self.assertTrue(pdf_storage.pdf_exists(response.document_id))
        self.assertEqual(pdf_storage.pdf_path(response.document_id).read_bytes(), b"%PDF-1.7 test")
        self.assertEqual(pdf_storage.pdf_sha256_path(response.document_id).read_text().strip(), pdf_storage.calculate_pdf_sha256(b"%PDF-1.7 test"))
        self.assertFalse(response.duplicate)
        store_chunks.assert_called_once_with(chunks)
        self.assertTrue(upload.closed)

    def test_post_index_failure_removes_new_pdf_and_does_not_report_success(self):
        upload = FakeUpload()
        with patch.object(main, "extract_and_chunk_pdf", return_value=(1, [])), \
                patch.object(main, "store_chunks", side_effect=RuntimeError("mock index failure")), \
                patch.object(main, "indexed_document_exists", return_value=False), \
                patch.object(main, "_find_document_with_pdf_sha256", return_value=None):
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(main.upload_document(upload))
        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(list((Path(self.temp_dir.name) / "pdfs").glob("*.pdf")), [])
        self.assertEqual(list((Path(self.temp_dir.name) / "pdfs").glob("*.sha256")), [])
        self.assertTrue(upload.closed)

    def test_uncertain_index_failure_retains_pdf_and_hash_for_safe_retry(self):
        with patch.object(main, "extract_and_chunk_pdf", return_value=(1, [])), \
                patch.object(main, "store_chunks", side_effect=RuntimeError("mock index failure")), \
                patch.object(main, "indexed_document_exists", side_effect=RuntimeError("mock lookup failure")), \
                patch.object(main, "_find_document_with_pdf_sha256", return_value=None):
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(main.upload_document(FakeUpload()))
        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(len(list((Path(self.temp_dir.name) / "pdfs").glob("*.pdf"))), 1)
        self.assertEqual(len(list((Path(self.temp_dir.name) / "pdfs").glob("*.sha256"))), 1)

    def test_put_attaches_legacy_pdf_without_reindexing_and_refuses_overwrite(self):
        document_id = str(uuid4())
        with patch.object(main, "indexed_document_exists", return_value=True), \
                patch.object(main, "extract_and_chunk_pdf") as extract, \
                patch.object(main, "store_chunks") as store_chunks:
            response = asyncio.run(main.attach_document_pdf(document_id, FakeUpload()))
            self.assertEqual(response.document_id, document_id)
            self.assertTrue(response.pdf_available)
            self.assertEqual(pdf_storage.pdf_sha256_path(document_id).read_text().strip(), pdf_storage.calculate_pdf_sha256(b"%PDF-1.7 test"))
            extract.assert_not_called()
            store_chunks.assert_not_called()

            with self.assertRaises(HTTPException) as raised:
                asyncio.run(main.attach_document_pdf(document_id, FakeUpload(b"%PDF-1.7 replacement")))
            self.assertEqual(raised.exception.status_code, 409)
            extract.assert_not_called()
            store_chunks.assert_not_called()

    def test_identical_reupload_reuses_existing_document_without_reindexing(self):
        document_id = str(uuid4())
        content = b"%PDF-1.7 identical content"
        pdf_storage.store_pdf(document_id, content)
        with patch.object(main, "list_indexed_documents", return_value=[{
            "document_id": document_id, "filename": "original.pdf", "chunk_count": 3,
        }]), patch.object(main, "extract_and_chunk_pdf") as extract, patch.object(main, "store_chunks") as store_chunks:
            response = asyncio.run(main.upload_document(FakeUpload(content, "renamed-copy.pdf")))

        self.assertEqual(response.document_id, document_id)
        self.assertEqual(response.filename, "original.pdf")
        self.assertEqual(response.status, "reused")
        self.assertTrue(response.duplicate)
        self.assertEqual(response.chunk_count, 3)
        self.assertEqual(response.page_count, 0)
        self.assertEqual(pdf_storage.pdf_sha256_path(document_id).read_text().strip(), pdf_storage.calculate_pdf_sha256(content))
        extract.assert_not_called()
        store_chunks.assert_not_called()

    def test_same_filename_with_different_bytes_is_indexed_as_new_document(self):
        existing_id = str(uuid4())
        existing_bytes = b"%PDF-1.7 first contents"
        new_bytes = b"%PDF-1.7 different contents"
        pdf_storage.store_pdf(existing_id, existing_bytes)
        with patch.object(main, "list_indexed_documents", return_value=[{
            "document_id": existing_id, "filename": "same.pdf", "chunk_count": 2,
        }]), patch.object(main, "extract_and_chunk_pdf", return_value=(1, [])) as extract, \
                patch.object(main, "store_chunks") as store_chunks:
            response = asyncio.run(main.upload_document(FakeUpload(new_bytes, "same.pdf")))

        self.assertNotEqual(response.document_id, existing_id)
        self.assertFalse(response.duplicate)
        extract.assert_called_once()
        store_chunks.assert_called_once()
        self.assertEqual(pdf_storage.pdf_path(existing_id).read_bytes(), existing_bytes)
        self.assertEqual(pdf_storage.pdf_path(response.document_id).read_bytes(), new_bytes)

    def test_legacy_pdf_without_sidecar_is_hashed_and_reused(self):
        document_id = str(uuid4())
        content = b"%PDF-1.7 legacy original"
        pdf_storage.store_pdf(document_id, content)
        with patch.object(main, "list_indexed_documents", return_value=[{
            "document_id": document_id, "filename": "legacy.pdf", "chunk_count": 4,
        }]), patch.object(main, "extract_and_chunk_pdf") as extract:
            response = asyncio.run(main.upload_document(FakeUpload(content)))

        self.assertTrue(response.duplicate)
        self.assertEqual(response.document_id, document_id)
        self.assertEqual(pdf_storage.pdf_sha256_path(document_id).read_text().strip(), pdf_storage.calculate_pdf_sha256(content))
        extract.assert_not_called()

    def test_stale_hash_mapping_without_pdf_or_indexed_document_is_removed(self):
        stale_id = str(uuid4())
        pdf_storage.store_pdf_sha256(stale_id, pdf_storage.calculate_pdf_sha256(b"%PDF-1.7 stale"))
        with patch.object(main, "list_indexed_documents", return_value=[]):
            self.assertIsNone(main._find_document_with_pdf_sha256(pdf_storage.calculate_pdf_sha256(b"%PDF-1.7 stale")))
        self.assertFalse(pdf_storage.pdf_sha256_exists(stale_id))

    def test_stale_digest_for_existing_pdf_is_recomputed_from_original_bytes(self):
        document_id = str(uuid4())
        existing_bytes = b"%PDF-1.7 actual legacy PDF bytes"
        stale_digest = pdf_storage.calculate_pdf_sha256(b"%PDF-1.7 different bytes")
        pdf_storage.store_pdf(document_id, existing_bytes)
        pdf_storage.store_pdf_sha256(document_id, stale_digest)
        with patch.object(main, "list_indexed_documents", return_value=[{
            "document_id": document_id, "filename": "legacy.pdf", "chunk_count": 2,
        }]):
            duplicate = main._find_document_with_pdf_sha256(stale_digest)
        self.assertIsNone(duplicate)
        self.assertEqual(pdf_storage.pdf_sha256_path(document_id).read_text().strip(), pdf_storage.calculate_pdf_sha256(existing_bytes))

    def test_hash_storage_failure_prevents_indexing_and_cleans_pdf(self):
        with patch.object(main, "_find_document_with_pdf_sha256", return_value=None), \
                patch.object(main, "extract_and_chunk_pdf", return_value=(1, [])), \
                patch.object(main, "store_pdf_sha256", side_effect=OSError("mock hash write failure")), \
                patch.object(main, "store_chunks") as store_chunks:
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(main.upload_document(FakeUpload()))
        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(list((Path(self.temp_dir.name) / "pdfs").glob("*.pdf")), [])
        store_chunks.assert_not_called()

    def test_duplicate_lookup_failure_does_not_extract_or_index(self):
        with patch.object(main, "_find_document_with_pdf_sha256", side_effect=OSError("mock lookup failure")), \
                patch.object(main, "extract_and_chunk_pdf") as extract, patch.object(main, "store_chunks") as store_chunks:
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(main.upload_document(FakeUpload()))
        self.assertEqual(raised.exception.status_code, 503)
        extract.assert_not_called()
        store_chunks.assert_not_called()

    def test_process_local_lock_serializes_concurrent_identical_uploads(self):
        content = b"%PDF-1.7 concurrent bytes"
        indexed = []
        with patch.object(main, "_find_document_with_pdf_sha256", side_effect=lambda digest: next((item for item in indexed if item["digest"] == digest), None)), \
                patch.object(main, "extract_and_chunk_pdf", side_effect=lambda _data, document_id, filename: (1, [{"text": "chunk", "metadata": {"document_id": document_id, "filename": filename}}])) as extract, \
                patch.object(main, "store_chunks", side_effect=lambda chunks: indexed.append({"document_id": chunks[0]["metadata"]["document_id"], "filename": "concurrent.pdf", "chunk_count": 1, "digest": pdf_storage.calculate_pdf_sha256(content)})):
            def upload_once():
                return asyncio.run(main.upload_document(FakeUpload(content, "concurrent.pdf")))

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                responses = list(executor.map(lambda _index: upload_once(), range(2)))

        self.assertEqual(sum(response.duplicate for response in responses), 1)
        self.assertEqual(responses[0].document_id, responses[1].document_id)
        extract.assert_called_once()

    def test_delete_cleans_hash_sidecar_with_pdf(self):
        document_id = str(uuid4())
        pdf_storage.store_pdf(document_id, b"%PDF-1.7 attached")
        pdf_storage.store_pdf_sha256(document_id, pdf_storage.calculate_pdf_sha256(b"%PDF-1.7 attached"))
        with patch.object(main, "indexed_document_exists", return_value=True), \
                patch.object(main, "delete_indexed_document", return_value=True):
            response = main.delete_document(document_id)
        self.assertTrue(response.deleted)
        self.assertFalse(pdf_storage.pdf_exists(document_id))
        self.assertFalse(pdf_storage.pdf_sha256_exists(document_id))

    def test_put_requires_existing_indexed_document(self):
        upload = FakeUpload()
        with patch.object(main, "indexed_document_exists", return_value=False), \
                patch.object(main, "store_pdf") as store:
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(main.attach_document_pdf(str(uuid4()), upload))
        self.assertEqual(raised.exception.status_code, 404)
        store.assert_not_called()
        self.assertTrue(upload.closed)

    def test_get_serves_pdf_inline_and_list_reports_availability(self):
        document_id = str(uuid4())
        pdf_storage.store_pdf(document_id, b"%PDF-1.7 saved")
        with patch.object(main, "indexed_document_exists", return_value=True):
            response = main.get_document_pdf(document_id)
        self.assertEqual(response.media_type, "application/pdf")
        self.assertIn("inline", response.headers["content-disposition"])
        self.assertEqual(Path(response.path), pdf_storage.pdf_path(document_id))

        with patch.object(main, "list_indexed_documents", return_value=[{
            "document_id": document_id, "filename": "test.pdf", "chunk_count": 1,
        }]):
            listed = main.list_documents()
        self.assertTrue(listed.documents[0].pdf_available)

    def test_get_rejects_invalid_uuid_and_missing_pdf(self):
        with self.assertRaises(HTTPException) as raised:
            main.get_document_pdf("../../outside")
        self.assertEqual(raised.exception.status_code, 422)
        with patch.object(main, "indexed_document_exists", return_value=True):
            with self.assertRaises(HTTPException) as raised:
                main.get_document_pdf(str(uuid4()))
        self.assertEqual(raised.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
