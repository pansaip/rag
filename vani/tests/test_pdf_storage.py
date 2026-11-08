import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from api import pdf_storage


class PdfStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_patch = patch.object(pdf_storage, "PDF_STORAGE_DIR", Path(self.temp_dir.name) / "pdfs")
        self.storage_patch.start()
        self.document_id = str(uuid4())

    def tearDown(self):
        self.storage_patch.stop()
        self.temp_dir.cleanup()

    def test_storage_path_uses_canonical_uuid_and_rejects_traversal(self):
        self.assertEqual(
            pdf_storage.pdf_path(self.document_id.upper()),
            Path(self.temp_dir.name) / "pdfs" / f"{self.document_id}.pdf",
        )
        for invalid in ("../../outside", "", "not-a-uuid"):
            with self.subTest(invalid=invalid), self.assertRaises(pdf_storage.PdfStorageError):
                pdf_storage.pdf_path(invalid)

    def test_atomic_store_and_no_overwrite(self):
        content = b"%PDF-1.7 test bytes"
        path = pdf_storage.store_pdf(self.document_id, content)
        self.assertEqual(path.read_bytes(), content)
        self.assertTrue(pdf_storage.pdf_exists(self.document_id))
        with self.assertRaises(pdf_storage.PdfAlreadyExists):
            pdf_storage.store_pdf(self.document_id, b"%PDF-1.7 different")
        self.assertEqual(path.read_bytes(), content)
        self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_rejects_bad_signature_and_oversized_pdf(self):
        with self.assertRaises(pdf_storage.PdfStorageError):
            pdf_storage.store_pdf(self.document_id, b"not a pdf")
        with patch.object(pdf_storage, "MAX_PDF_BYTES", 8):
            with self.assertRaises(pdf_storage.PdfStorageError):
                pdf_storage.store_pdf(self.document_id, b"%PDF-1234")
        self.assertFalse(pdf_storage.PDF_STORAGE_DIR.exists())


if __name__ == "__main__":
    unittest.main()
