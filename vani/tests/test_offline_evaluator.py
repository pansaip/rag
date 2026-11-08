import json
import tempfile
import unittest
from pathlib import Path

from evaluation.evaluator import evaluate, load_dataset, load_trace, render_markdown


def make_case(case_id, *, answerable=True, support=None, document_id="doc-a"):
    case = {
        "case_id": case_id,
        "question": "question",
        "answer_criteria": "human label only",
        "answerable": answerable,
        "language": "en",
        "category": "specific_factual",
    }
    case["document_id"] = document_id
    if support is not None:
        case["expected_support"] = support
    return case


def make_result(case_id, *, selected="doc-a", retrieval=None, supplied=None, citations=None, prediction=True, latency=10):
    return {
        "case_id": case_id,
        "selected_document_id": selected,
        "retrieval": retrieval or [],
        "provided_source_ids": supplied or [],
        "citations": citations or [],
        "predicted_answerable": prediction,
        "latency_ms": latency,
        "answer": "unscored answer text",
    }


class OfflineEvaluatorTests(unittest.TestCase):
    def test_real_pdf_dataset_uses_actual_document_and_avoids_unverified_chunk_labels(self):
        path = Path(__file__).resolve().parents[1] / "evaluation" / "real_dataset.json"
        dataset = load_dataset(path)
        cases = dataset["cases"]
        self.assertEqual(len(cases), 14)
        self.assertTrue(all(case.get("document_id") == "12cc259a-5f4f-479d-a4c4-0a1583d67a51" for case in cases))
        self.assertEqual(sum(case["label_status"] == "verified_pdf_fact" for case in cases), 12)
        self.assertEqual(sum(case["label_status"].startswith("provisional_unanswerable") for case in cases), 2)
        self.assertTrue(all("chunk_indices" not in (case.get("expected_support") or {}) for case in cases))
        self.assertEqual(sum(not case["answerable"] for case in cases), 2)
        self.assertEqual({case["language"] for case in cases}, {"en", "th"})

    def test_metrics_distinguish_chunk_hit_recall_page_and_citations(self):
        dataset = {
            "name": "metric-test",
            "cases": [
                make_case("labeled", support={"pages": [1], "chunk_indices": [2, 3]}),
                make_case("unanswerable", answerable=False),
            ],
        }
        trace = {
            "labeled": make_result(
                "labeled",
                retrieval=[
                    {"document_id": "doc-a", "page_number": 2, "chunk_index": 2, "source_id": "S1"},
                    {"document_id": "doc-a", "page_number": 1, "chunk_index": 9, "source_id": "S2"},
                ],
                supplied=["S1"],
                citations=[
                    {"source_id": "S1"},
                    {"source_id": "S9", "document_id": "doc-a", "page_number": 1},
                ],
                prediction=True,
                latency=10,
            ),
            "unanswerable": make_result("unanswerable", prediction=False, latency=30),
        }
        report = evaluate(dataset, trace, k=2)
        metrics = report["metrics"]
        self.assertEqual(metrics["retrieval_hit_rate_at_k"]["value"], 1.0)
        self.assertEqual(metrics["retrieval_hit_rate_at_k"]["denominator"], 1)
        self.assertEqual(metrics["chunk_recall_at_k"]["value"], 0.5)
        self.assertEqual(metrics["chunk_recall_at_k"]["denominator"], 2)
        self.assertEqual(metrics["page_hit_rate_at_k"]["value"], 1.0)
        self.assertEqual(metrics["supporting_page_accuracy"]["value"], 0.0)
        self.assertEqual(metrics["citation_id_validity"]["value"], 0.5)
        self.assertEqual(metrics["answerability_accuracy"]["value"], 1.0)
        self.assertEqual(metrics["response_latency_ms"]["mean"], 20)
        self.assertIn("does not score semantic answer correctness", " ".join(report["limitations"]))

    def test_missing_labels_and_missing_traces_are_reported_not_scored(self):
        dataset = {"cases": [make_case("unlabeled"), make_case("no-trace", support={"pages": [4]})]}
        report = evaluate(dataset, {"unlabeled": make_result("unlabeled")}, k=5)
        self.assertIsNone(report["metrics"]["retrieval_hit_rate_at_k"]["value"])
        self.assertEqual(report["metrics"]["retrieval_hit_rate_at_k"]["missing_label_cases"], 2)
        self.assertIsNone(report["metrics"]["page_hit_rate_at_k"]["value"])
        self.assertEqual(report["sample_counts"]["missing_trace_cases"], 1)
        self.assertEqual(report["metrics"]["response_latency_ms"]["sample_count"], 1)

    def test_empty_dataset_produces_empty_report(self):
        report = evaluate({"cases": []}, {}, k=5)
        self.assertEqual(report["sample_counts"]["dataset_cases"], 0)
        self.assertIsNone(report["metrics"]["chunk_recall_at_k"]["value"])
        self.assertEqual(report["metrics"]["response_latency_ms"]["sample_count"], 0)
        self.assertIn("| chunk_recall_at_k | N/A | 0 / 0 |", render_markdown(report))

    def test_retrieval_only_trace_keeps_chat_metrics_unavailable(self):
        dataset = {"cases": [make_case("retrieval-only", support={"pages": [1]})]}
        retrieval_only = {
            "selected_document_id": "doc-a",
            "retrieval": [{"document_id": "doc-a", "page_number": 1, "chunk_index": 0}],
            "retrieval_path": "vector_search_top10",
        }
        report = evaluate(dataset, {"retrieval-only": retrieval_only})
        metrics = report["metrics"]
        self.assertEqual(metrics["page_hit_rate_at_k"]["value"], 1.0)
        self.assertIsNone(metrics["citation_id_validity"]["value"])
        self.assertEqual(metrics["citation_id_validity"]["missing_label_cases"], 1)
        self.assertIsNone(metrics["supporting_page_accuracy"]["value"])
        self.assertEqual(metrics["supporting_page_accuracy"]["missing_label_cases"], 1)
        self.assertIsNone(metrics["answerability_accuracy"]["value"])

    def test_document_scope_prevents_other_document_chunks_from_counting(self):
        dataset = {"cases": [make_case("isolated", support={"chunk_indices": [0]}, document_id="doc-a")]}
        trace = {
            "isolated": make_result(
                "isolated",
                selected="doc-a",
                retrieval=[{"document_id": "doc-b", "page_number": 1, "chunk_index": 0}],
            )
        }
        report = evaluate(dataset, trace)
        self.assertEqual(report["metrics"]["retrieval_hit_rate_at_k"]["value"], 0.0)
        self.assertEqual(report["metrics"]["chunk_recall_at_k"]["value"], 0.0)

    def test_dataset_and_trace_loaders_reject_malformed_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "dataset.json"
            trace_path = Path(temp_dir) / "trace.json"
            dataset_path.write_text(json.dumps({"cases": [{"case_id": "x"}]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "question"):
                load_dataset(dataset_path)

            dataset_path.write_text(json.dumps({"cases": [make_case("x")]}), encoding="utf-8")
            load_dataset(dataset_path)
            trace_path.write_text(json.dumps({"results": [{"case_id": "x", "retrieval": "bad"}]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "selected_document_id"):
                load_trace(trace_path, {"x"})


if __name__ == "__main__":
    unittest.main()
