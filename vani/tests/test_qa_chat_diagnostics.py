import unittest
from unittest.mock import patch
from uuid import uuid4

from api import qa


DOCUMENT_ID = str(uuid4())
SOURCE = {
    "document_id": DOCUMENT_ID,
    "filename": "private-paper.pdf",
    "page_number": 1,
    "chunk_index": 0,
    "text": "PRIVATE DOCUMENT PASSAGE",
}


class FakeClient:
    def __init__(self, output_text=None, error=None):
        self.responses = self
        self.output_text = output_text
        self.error = error

    def create(self, **_kwargs):
        if self.error:
            raise self.error
        return type("Response", (), {"output_text": self.output_text})()


class ChatDiagnosticsTests(unittest.TestCase):
    def run_with_diagnostics(self, question, chunks=None, output="Answer [S1].", error=None, document_id=DOCUMENT_ID):
        client = FakeClient(output, error)
        with patch.dict("os.environ", {
            "VANI_ENABLE_LOCAL_CHAT_DIAGNOSTICS": "1",
            "GROQ_API_KEY": "mock-secret",
            "GROQ_MODEL": "mock-model",
        }), patch.object(qa, "search_chunks", return_value=chunks or []), \
                patch.object(qa, "diagnostic_first_page_chunks", return_value=chunks or []), \
                patch.object(qa, "_get_openai_client", return_value=client):
            return qa.answer_question(question, document_id)

    def test_missing_selection_reason_and_safe_fields(self):
        with patch.dict("os.environ", {"VANI_ENABLE_LOCAL_CHAT_DIAGNOSTICS": "1"}, clear=True):
            with self.assertLogs(qa.logger, level="INFO") as captured:
                qa.answer_question("Who wrote this paper?")
        message = captured.output[0]
        self.assertIn("intent=authors", message)
        self.assertIn("document_id_provided=False document_id_validated=False", message)
        self.assertIn("fallback_reason=missing_document_selection", message)
        self.assertNotIn("Who wrote this paper", message)

    def test_empty_retrieval_reason(self):
        with patch.dict("os.environ", {"VANI_ENABLE_LOCAL_CHAT_DIAGNOSTICS": "1"}, clear=True), \
                patch.object(qa, "search_chunks", return_value=[]):
            with self.assertLogs(qa.logger, level="INFO") as captured:
                qa.answer_question("deepsea คืออะไร", DOCUMENT_ID)
        self.assertIn("retrieval_path=vector retrieved_chunk_count=0 retrieved_locations=none source_count=0", captured.output[0])
        self.assertIn("fallback_reason=empty_retrieval", captured.output[0])
        self.assertIn("document_id_validated=True", captured.output[0])

    def test_document_isolation_failure_reason(self):
        mixed = [SOURCE, {**SOURCE, "document_id": str(uuid4()), "chunk_index": 1}]
        with patch.dict("os.environ", {"VANI_ENABLE_LOCAL_CHAT_DIAGNOSTICS": "1"}, clear=True), \
                patch.object(qa, "search_chunks", return_value=mixed):
            with self.assertLogs(qa.logger, level="INFO") as captured:
                qa.answer_question("deepsea คืออะไร")
        self.assertIn("fallback_reason=document_isolation_failure", captured.output[0])
        self.assertIn("retrieved_chunk_count=2 retrieved_locations=1:0,1:1 source_count=0", captured.output[0])

    def test_empty_model_response_reason(self):
        with self.assertRaises(qa.RAGServiceError), self.assertLogs(qa.logger, level="INFO") as captured:
            self.run_with_diagnostics("deepsea คืออะไร", [SOURCE], output="")
        self.assertIn("llm_attempted=True llm_nonempty=False", captured.output[0])
        self.assertIn("fallback_reason=empty_model_response", captured.output[0])

    def test_unknown_citations_are_removed_without_fallback(self):
        with self.assertLogs(qa.logger, level="INFO") as captured:
            result = self.run_with_diagnostics("deepsea คืออะไร", [SOURCE], output="Some answer [S9].")
        self.assertEqual(result["citations"], [])
        self.assertIn("citation_marker_count=1 valid_citation_id_count=0", captured.output[0])
        self.assertIn("fallback_reason=none", captured.output[0])
        self.assertEqual(result["answer"], "Some answer.")

    def test_provider_error_logs_model_error_without_exception_text_or_secrets(self):
        with self.assertRaises(qa.RAGServiceError), self.assertLogs(qa.logger, level="INFO") as captured:
            self.run_with_diagnostics("deepsea คืออะไร", [SOURCE], error=RuntimeError("mock-secret PRIVATE DOCUMENT PASSAGE"))
        message = " ".join(captured.output)
        self.assertIn("llm_attempted=True", message)
        self.assertIn("fallback_reason=model_error", message)
        self.assertNotIn("mock-secret", message)
        self.assertNotIn("PRIVATE DOCUMENT PASSAGE", message)

    def test_success_log_counts_citations_and_omits_content(self):
        with self.assertLogs(qa.logger, level="INFO") as captured:
            result = self.run_with_diagnostics("deepsea คืออะไร", [SOURCE], output="Supported [ s1 ] and [S1].")
        message = captured.output[0]
        self.assertEqual(len(result["citations"]), 1)
        self.assertIn("retrieved_chunk_count=1 retrieved_locations=1:0 source_count=1", message)
        self.assertIn("llm_attempted=True llm_nonempty=True", message)
        self.assertIn("citation_marker_count=2 valid_citation_id_count=1", message)
        self.assertIn("fallback_reason=none", message)
        self.assertNotIn("PRIVATE DOCUMENT PASSAGE", message)
        self.assertNotIn("Supported", message)

    def test_logging_is_disabled_without_opt_in(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertNoLogs(qa.logger, level="INFO"):
                qa.answer_question("Who wrote this paper?")

    def test_retrieval_exception_is_reported_as_other(self):
        with patch.dict("os.environ", {"VANI_ENABLE_LOCAL_CHAT_DIAGNOSTICS": "1"}, clear=True), \
                patch.object(qa, "search_chunks", side_effect=RuntimeError("private retrieval detail")):
            with self.assertRaises(qa.RAGServiceError), self.assertLogs(qa.logger, level="INFO") as captured:
                qa.answer_question("deepsea คืออะไร", DOCUMENT_ID)
        self.assertIn("fallback_reason=other", captured.output[0])
        self.assertNotIn("private retrieval detail", captured.output[0])


if __name__ == "__main__":
    unittest.main()
