import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from api import main, observability, qa


DOCUMENT_ID = "f34a8f72-8d92-4c24-b654-ea9a1ab3dd03"
CHUNK = {
    "document_id": DOCUMENT_ID,
    "filename": "private-filename.pdf",
    "page_number": 2,
    "chunk_index": 0,
    "text": "PRIVATE PASSAGE that must never be written.",
}


class FakeResponses:
    def __init__(self, response):
        self.response = response

    def create(self, **_kwargs):
        return self.response


class FakeClient:
    def __init__(self, response):
        self.responses = FakeResponses(response)


class ObservabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.trace_file = Path(self.temp_dir.name) / "nested" / "traces.jsonl"
        self.path_patch = patch.object(observability, "TRACE_PATH", self.trace_file)
        self.path_patch.start()

    def tearDown(self):
        self.path_patch.stop()
        self.temp_dir.cleanup()

    def read_records(self):
        return [json.loads(line) for line in self.trace_file.read_text(encoding="utf-8").splitlines()]

    def test_default_off_does_not_create_trace_file(self):
        with patch.dict("os.environ", {}, clear=True):
            with observability.chat_request("test-model"):
                observability.mark_success(1)
        self.assertFalse(self.trace_file.exists())

    def test_enabled_writes_expected_fields_without_sensitive_content(self):
        with patch.dict("os.environ", {observability.ENABLE_ENV: "1"}, clear=True):
            with observability.chat_request("groq/test-model"):
                observability.record_retrieval("vector", 1, 2.5)
                observability.record_llm(3.5, SimpleNamespace(usage=SimpleNamespace(input_tokens=11, output_tokens=7, total_tokens=18)))
                observability.mark_success(1)
        record = self.read_records()[0]
        self.assertEqual(record["retrieval_path"], "vector")
        self.assertEqual(record["retrieved_chunk_count"], 1)
        self.assertEqual(record["retrieval_duration_ms"], 2.5)
        self.assertEqual(record["llm_duration_ms"], 3.5)
        self.assertEqual((record["prompt_tokens"], record["completion_tokens"], record["total_tokens"]), (11, 7, 18))
        self.assertEqual(record["citations_returned"], 1)
        self.assertEqual(record["outcome"], "success")
        self.assertEqual(set(record), {
            "timestamp_utc", "request_id", "overall_duration_ms", "retrieval_duration_ms", "llm_duration_ms",
            "retrieval_path", "retrieved_chunk_count", "citations_returned", "used_fallback", "fallback_reason",
            "outcome", "model", "prompt_tokens", "completion_tokens", "total_tokens",
        })

    def test_missing_usage_is_null_and_fallback_reason_is_recorded(self):
        with patch.dict("os.environ", {observability.ENABLE_ENV: "1"}, clear=True):
            with observability.chat_request("model"):
                observability.record_llm(1.2, SimpleNamespace(output_text="ok"))
                observability.mark_fallback("empty_retrieval")
        record = self.read_records()[0]
        self.assertEqual(record["outcome"], "fallback")
        self.assertTrue(record["used_fallback"])
        self.assertEqual(record["fallback_reason"], "empty_retrieval")
        self.assertIsNone(record["prompt_tokens"])
        self.assertIsNone(record["completion_tokens"])
        self.assertIsNone(record["total_tokens"])

    def test_errors_are_recorded_without_exception_details(self):
        with patch.dict("os.environ", {observability.ENABLE_ENV: "1"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "private"):
                with observability.chat_request("model"):
                    observability.mark_error("model_error")
                    raise RuntimeError("private exception and secret-token")
        raw = self.trace_file.read_text(encoding="utf-8")
        self.assertNotIn("private", raw)
        self.assertNotIn("secret-token", raw)
        self.assertEqual(self.read_records()[0]["outcome"], "error")
        self.assertEqual(self.read_records()[0]["fallback_reason"], "model_error")

    def test_concurrent_requests_append_complete_json_lines(self):
        with patch.dict("os.environ", {observability.ENABLE_ENV: "1"}, clear=True):
            def worker():
                with observability.chat_request("model"):
                    observability.record_retrieval("first_page", 2, 0.1)
            threads = [threading.Thread(target=worker) for _ in range(20)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        records = self.read_records()
        self.assertEqual(len(records), 20)
        self.assertEqual(len({record["request_id"] for record in records}), 20)

    def test_write_failure_does_not_escape_or_break_chat(self):
        with patch.dict("os.environ", {observability.ENABLE_ENV: "1"}, clear=True), \
                patch.object(observability.Path, "open", side_effect=OSError("private filesystem details")):
            with observability.chat_request("model"):
                observability.mark_success(0)

    def test_chat_trace_records_retrieval_and_tokens_without_changing_response(self):
        response = SimpleNamespace(
            output_text="Grounded answer [S1].",
            usage=SimpleNamespace(input_tokens=20, output_tokens=5, total_tokens=25),
        )
        with patch.dict("os.environ", {
            observability.ENABLE_ENV: "1", "GROQ_API_KEY": "secret-key", "GROQ_MODEL": "mock-model",
        }, clear=True), patch.object(qa, "search_chunks", return_value=[CHUNK]), \
                patch.object(qa, "_get_openai_client", return_value=FakeClient(response)):
            with observability.chat_request("mock-model"):
                result = qa.answer_question("private user question", DOCUMENT_ID)
        self.assertEqual(result["answer"], "Grounded answer [S1].")
        self.assertEqual(len(result["citations"]), 1)
        raw = self.trace_file.read_text(encoding="utf-8")
        for sensitive in ("private user question", "Grounded answer", "PRIVATE PASSAGE", "private-filename", DOCUMENT_ID, "secret-key"):
            self.assertNotIn(sensitive, raw)
        record = self.read_records()[0]
        self.assertEqual(record["retrieval_path"], "vector")
        self.assertEqual(record["retrieved_chunk_count"], 1)
        self.assertIsInstance(record["retrieval_duration_ms"], (int, float))
        self.assertGreaterEqual(record["retrieval_duration_ms"], 0)
        self.assertIsInstance(record["llm_duration_ms"], (int, float))
        self.assertGreaterEqual(record["llm_duration_ms"], 0)
        self.assertEqual(record["citations_returned"], 1)
        self.assertEqual((record["prompt_tokens"], record["completion_tokens"], record["total_tokens"]), (20, 5, 25))

    def test_chat_endpoint_keeps_response_schema_unchanged(self):
        expected = {"answer": "plain answer", "citations": []}
        with patch.dict("os.environ", {observability.ENABLE_ENV: "1"}, clear=True), \
                patch.object(main, "answer_question", return_value=expected):
            response = main.chat(main.ChatRequest(question="a question"))
        self.assertEqual(response.model_dump(), expected)
        self.assertEqual(self.read_records()[0]["outcome"], "success")

    def test_chat_empty_retrieval_and_provider_error_outcomes(self):
        with patch.dict("os.environ", {observability.ENABLE_ENV: "1"}, clear=True), \
                patch.object(qa, "search_chunks", return_value=[]):
            with observability.chat_request("mock-model"):
                fallback = qa.answer_question("deepsea definition?")
        self.assertEqual(fallback["citations"], [])
        self.assertEqual(self.read_records()[0]["outcome"], "fallback")
        self.assertEqual(self.read_records()[0]["fallback_reason"], "empty_retrieval")

        self.trace_file.unlink()
        with patch.dict("os.environ", {
            observability.ENABLE_ENV: "1", "GROQ_API_KEY": "secret-key", "GROQ_MODEL": "mock-model",
        }, clear=True), patch.object(qa, "search_chunks", return_value=[CHUNK]), \
                patch.object(qa, "_get_openai_client", return_value=FakeClient(None)):
            # Simulate a provider failure while keeping exception text private.
            qa._get_openai_client("secret-key").responses.create = lambda **_kw: (_ for _ in ()).throw(RuntimeError("private provider response"))
            with observability.chat_request("mock-model"):
                with self.assertRaises(qa.RAGServiceError):
                    qa.answer_question("private question", DOCUMENT_ID)
        raw = self.trace_file.read_text(encoding="utf-8")
        self.assertNotIn("private provider response", raw)
        self.assertNotIn("secret-key", raw)
        self.assertEqual(self.read_records()[0]["outcome"], "error")
        self.assertEqual(self.read_records()[0]["fallback_reason"], "model_error")


if __name__ == "__main__":
    unittest.main()
