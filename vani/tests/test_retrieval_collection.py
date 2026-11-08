import json
import unittest
from io import BytesIO
from urllib.error import HTTPError

from evaluation.collect_retrieval import DOCUMENT_ID, CollectionError, collect_baselines
from evaluation.evaluator import load_trace


def response_json(value):
    return FakeResponse(json.dumps(value).encode("utf-8"))


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body


def make_dataset(questions):
    return {
        "cases": [
            {
                "case_id": case_id,
                "question": question,
                "document_id": DOCUMENT_ID,
                "expected_answer": "criteria text",
                "expected_support": {"pages": [1]},
                "answerable": True,
                "language": "en",
                "category": "specific_factual",
            }
            for case_id, question in questions
        ]
    }


def diagnostic_body():
    return {
        "document_id": DOCUMENT_ID,
        "first_page_chunks": [
            {
                "document_id": DOCUMENT_ID,
                "page_number": 1,
                "chunk_index": index,
                "score": None,
                "excerpt": f"PRIVATE CHUNK TEXT {index}",
            }
            for index in range(20, 26)
        ],
        "first_page_chunk_count": 6,
        "first_page_truncated": False,
        "questions": [],
        "interpretation": "diagnostic only",
    }


def search_body(document_id=DOCUMENT_ID, indices=range(10)):
    return {
        "results": [
            {
                "document_id": document_id,
                "filename": "private.pdf",
                "page_number": 1 + index // 3,
                "chunk_index": index,
                "score": 0.9 - index / 100,
                "text": f"PRIVATE VECTOR TEXT {index}",
            }
            for index in indices
        ]
    }


class RetrievalCollectionTests(unittest.TestCase):
    def test_trace_schema_separates_vector_and_chat_paths_without_text_or_fabricated_ids(self):
        dataset = make_dataset([
            ("authors", "Who wrote this paper?"),
            ("overview", "What is this paper about?"),
            ("specific", "What is DEEP SEA?"),
        ])
        calls = []

        def opener(request, timeout):
            calls.append((request.full_url, request.get_method(), timeout))
            if request.full_url.endswith("/health"):
                return response_json({"status": "ok"})
            if "/diagnostics/retrieval/" in request.full_url:
                return response_json(diagnostic_body())
            return response_json(search_body())

        collection = collect_baselines(dataset, opener=opener)
        vector = collection["vector_trace"]
        chat = collection["chat_trace"]
        self.assertEqual(len(vector["results"]), 3)
        self.assertEqual(len(vector["results"][0]["retrieval"]), 10)
        self.assertEqual(vector["results"][0]["retrieval_path"], "vector_search_top10")
        self.assertEqual(chat["results"][0]["retrieval_path"], "first_page")
        self.assertEqual(chat["results"][1]["retrieval_path"], "first_page_plus_vector")
        self.assertEqual(len(chat["results"][1]["retrieval"]), 10)
        self.assertEqual(chat["results"][2]["retrieval_path"], "vector")
        self.assertNotIn("PRIVATE", json.dumps(collection))
        self.assertNotIn("source_id", json.dumps(collection))
        self.assertNotIn("citations", vector["results"][0])
        self.assertTrue(all("/chat" not in call[0] for call in calls))
        loaded = load_trace_file(vector, {case["case_id"] for case in dataset["cases"]})
        self.assertEqual(len(loaded), 3)

    def test_exact_document_isolation_rejects_foreign_search_results(self):
        dataset = make_dataset([("case", "A specific question?")])

        def opener(request, timeout):
            if request.full_url.endswith("/health"):
                return response_json({"status": "ok"})
            if "/diagnostics/retrieval/" in request.full_url:
                return response_json(diagnostic_body())
            return response_json(search_body(document_id="another-document"))

        collection = collect_baselines(dataset, opener=opener)
        self.assertEqual(collection["vector_trace"]["results"], [])
        self.assertEqual(collection["vector_trace"]["failed_cases"][0]["error_type"], "document_isolation_violation")
        self.assertNotIn("another-document", json.dumps(collection))

    def test_exact_document_isolation_rejects_foreign_first_page_diagnostic(self):
        dataset = make_dataset([("authors", "Who wrote this paper?")])

        def opener(request, timeout):
            if request.full_url.endswith("/health"):
                return response_json({"status": "ok"})
            if "/diagnostics/retrieval/" in request.full_url:
                body = diagnostic_body()
                body["document_id"] = "another-document"
                return response_json(body)
            return response_json(search_body())

        collection = collect_baselines(dataset, opener=opener)
        self.assertEqual(len(collection["vector_trace"]["results"]), 1)
        self.assertEqual(collection["chat_trace"]["results"], [])
        self.assertEqual(collection["chat_trace"]["failed_cases"][0]["error_type"], "document_isolation_violation")
        self.assertNotIn("another-document", json.dumps(collection))

    def test_empty_search_results_are_recorded_as_empty_not_fabricated(self):
        dataset = make_dataset([("empty", "A specific question?")])

        def opener(request, timeout):
            if request.full_url.endswith("/health"):
                return response_json({"status": "ok"})
            if "/diagnostics/retrieval/" in request.full_url:
                raise HTTPError(request.full_url, 404, "disabled", {}, BytesIO(b""))
            return response_json(search_body(indices=[]))

        collection = collect_baselines(dataset, opener=opener)
        self.assertEqual(len(collection["vector_trace"]["results"]), 1)
        self.assertEqual(collection["vector_trace"]["results"][0]["retrieval"], [])
        self.assertEqual(collection["vector_trace"]["failed_cases"], [])
        self.assertEqual(collection["chat_trace"]["results"][0]["retrieval"], [])

    def test_missing_results_and_partial_http_failures_are_reported_without_substitution(self):
        dataset = make_dataset([
            ("fails", "Question one?"),
            ("succeeds", "Question two?"),
        ])

        def opener(request, timeout):
            if request.full_url.endswith("/health"):
                return response_json({"status": "ok"})
            if "/diagnostics/retrieval/" in request.full_url:
                raise HTTPError(request.full_url, 404, "disabled", {}, BytesIO(b"do not log body"))
            payload = json.loads(request.data.decode("utf-8"))
            if payload["question"] == "Question one?":
                raise HTTPError(request.full_url, 503, "failure", {}, BytesIO(b"private body"))
            return response_json(search_body())

        collection = collect_baselines(dataset, opener=opener)
        self.assertEqual(len(collection["vector_trace"]["results"]), 1)
        self.assertEqual(collection["vector_trace"]["failed_cases"][0]["http_status"], 503)
        self.assertEqual(collection["chat_trace"]["failed_cases"][0]["error_type"], "vector_results_unavailable")
        self.assertFalse(collection["chat_trace"]["first_page_diagnostics_available"])
        self.assertEqual(len(collection["chat_trace"]["results"]), 1)
        self.assertEqual(collection["diagnostic_failure"]["http_status"], 404)
        self.assertNotIn("private body", json.dumps(collection))

    def test_local_base_url_is_required_before_network_access(self):
        dataset = make_dataset([("case", "Question?")])
        called = False

        def opener(_request, timeout):
            nonlocal called
            called = True
            return response_json({"status": "ok"})

        with self.assertRaisesRegex(CollectionError, "non_local_base_url"):
            collect_baselines(dataset, base_url="https://example.com", opener=opener)
        self.assertFalse(called)


def load_trace_file(trace, case_ids):
    # Exercise the same in-memory trace contract without writing output files.
    from pathlib import Path
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        path = Path(directory) / "trace.json"
        path.write_text(json.dumps(trace), encoding="utf-8")
        return load_trace(path, case_ids)


if __name__ == "__main__":
    unittest.main()
