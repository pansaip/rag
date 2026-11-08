import json
import tempfile
import unittest
from pathlib import Path

from evaluation import summarize_observability as summary_tool


class ObservabilitySummaryTests(unittest.TestCase):
    def test_empty_file_has_zero_denominators_and_null_distributions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "empty.jsonl"
            path.write_text("", encoding="utf-8")
            summary = summary_tool.read_trace_file(path)
        self.assertEqual(summary["counts"]["total_lines"], 0)
        self.assertEqual(summary["counts"]["valid_json_records"], 0)
        self.assertIsNone(summary["fallback_rate"]["rate"])
        self.assertIsNone(summary["citation_presence_rate"]["rate"])
        self.assertIsNone(summary["latency_ms"]["overall_duration_ms"]["average"])
        self.assertEqual(summary["latency_ms"]["overall_duration_ms"]["missing_count"], 0)

    def test_latency_percentiles_and_rates_use_observed_denominators(self):
        rows = [
            {"outcome": "success", "retrieval_path": "vector", "used_fallback": False,
             "citations_returned": 1, "overall_duration_ms": 10, "retrieval_duration_ms": 2,
             "llm_duration_ms": 7, "prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
            {"outcome": "fallback", "retrieval_path": "first_page", "used_fallback": True,
             "citations_returned": 0, "overall_duration_ms": 20, "retrieval_duration_ms": 4,
             "llm_duration_ms": None, "prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
            {"outcome": "error", "retrieval_path": "none", "used_fallback": None,
             "citations_returned": None, "overall_duration_ms": 30, "retrieval_duration_ms": None,
             "llm_duration_ms": 9, "prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55},
        ]
        summary = summary_tool.summarize_records(rows)
        self.assertEqual(summary["requests_by_outcome"], {"success": 1, "fallback": 1, "error": 1, "unknown": 0})
        self.assertEqual(summary["requests_by_retrieval_path"]["vector"], 1)
        self.assertEqual(summary["latency_ms"]["overall_duration_ms"]["average"], 20)
        self.assertEqual(summary["latency_ms"]["overall_duration_ms"]["median"], 20)
        self.assertEqual(summary["latency_ms"]["overall_duration_ms"]["p95"], 30)
        self.assertEqual(summary["latency_ms"]["llm_duration_ms"]["missing_count"], 1)
        self.assertEqual(summary["fallback_rate"], {
            "fallback_requests": 1, "observed_requests": 2, "missing_count": 1, "rate": 0.5,
        })
        self.assertEqual(summary["citation_presence_rate"], {
            "requests_with_one_or_more_citations": 1, "observed_requests": 2, "missing_count": 1, "rate": 0.5,
        })
        self.assertEqual(summary["token_usage"]["prompt_tokens"]["sum"], 150)
        self.assertEqual(summary["token_usage"]["total_tokens"]["missing_count"], 1)

    def test_malformed_and_non_object_lines_are_counted_and_skipped(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "traces.jsonl"
            path.write_text('{"outcome":"success"}\n{broken\n[1,2]\nnull\n', encoding="utf-8")
            summary = summary_tool.read_trace_file(path)
        self.assertEqual(summary["counts"]["total_lines"], 4)
        self.assertEqual(summary["counts"]["valid_json_records"], 1)
        self.assertEqual(summary["counts"]["malformed_json_lines"], 1)
        self.assertEqual(summary["counts"]["non_object_json_lines"], 2)
        self.assertEqual(summary["counts"]["incomplete_records"], 1)
        self.assertEqual(summary["requests_by_retrieval_path"]["unknown"], 1)
        self.assertEqual(summary["latency_ms"]["overall_duration_ms"]["missing_count"], 1)

    def test_invalid_numeric_values_are_missing_not_reported(self):
        summary = summary_tool.summarize_records([
            {"overall_duration_ms": -1, "retrieval_duration_ms": float("nan"),
             "llm_duration_ms": True, "prompt_tokens": 1.5, "completion_tokens": -2,
             "total_tokens": True, "citations_returned": True, "used_fallback": "false",
             "outcome": "secret answer", "retrieval_path": "private filename"}
        ])
        self.assertEqual(summary["requests_by_outcome"]["unknown"], 1)
        self.assertEqual(summary["requests_by_retrieval_path"]["unknown"], 1)
        self.assertEqual(summary["latency_ms"]["overall_duration_ms"]["missing_count"], 1)
        self.assertEqual(summary["token_usage"]["prompt_tokens"]["missing_count"], 1)
        self.assertNotIn("secret answer", json.dumps(summary))
        self.assertNotIn("private filename", json.dumps(summary))

    def test_markdown_and_json_reports_contain_aggregates_only(self):
        summary = summary_tool.summarize_records([
            {"outcome": "success", "retrieval_path": "vector", "used_fallback": False,
             "citations_returned": 0, "overall_duration_ms": 1, "retrieval_duration_ms": 0.2,
             "llm_duration_ms": 0.5, "prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6,
             "request_id": "private-request-id", "model": "private-model"}
        ])
        markdown = summary_tool.render_markdown(summary)
        serialized = json.dumps(summary)
        self.assertIn("Fallback rate", markdown)
        self.assertIn("Citation presence rate", markdown)
        self.assertIn("Citation presence is not citation validity", markdown)
        self.assertNotIn("private-request-id", markdown + serialized)
        self.assertNotIn("private-model", markdown + serialized)

    def test_cli_refuses_to_overwrite_input_trace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.jsonl"
            path.write_text("", encoding="utf-8")
            with self.assertRaises(SystemExit) as raised:
                summary_tool.main(["--input", str(path), "--json-out", str(path), "--markdown-out", str(Path(temp_dir) / "report.md")])
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
