"""Offline, dependency-free metrics for recorded or mocked VANI RAG traces."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any


LIMITATIONS = [
    "This evaluator does not score semantic answer correctness, completeness, or faithfulness.",
    "Answerability accuracy uses the explicitly supplied predicted_answerable trace label; it does not infer answerability from answer text.",
    "Chunk relevance uses document-scoped chunk_index labels; a page match is reported separately and is never counted as a chunk match.",
    "Illustrative example labels are not ground truth for arbitrary PDFs. Replace them with human-verified labels before interpreting scores.",
    "All metrics use supplied files only; the runner does not call an LLM or connect to Qdrant.",
]


def _load_json(path: Path, kind: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {kind} JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{kind} must be a JSON object")
    return value


def load_dataset(path: Path) -> dict[str, Any]:
    dataset = _load_json(path, "dataset")
    cases = dataset.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Dataset must contain a cases array")
    seen: set[str] = set()
    for index, case in enumerate(cases):
        prefix = f"Dataset case {index}"
        if not isinstance(case, dict):
            raise ValueError(f"{prefix} must be an object")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen:
            raise ValueError(f"{prefix} needs a unique non-empty case_id")
        seen.add(case_id)
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError(f"{prefix} needs a non-empty question")
        if (isinstance(case.get("document_id"), str) and case["document_id"].strip()) == (
            isinstance(case.get("document_fixture"), str) and bool(case["document_fixture"].strip())
        ):
            raise ValueError(f"{prefix} must set exactly one of document_id or document_fixture")
        if not isinstance(case.get("answerable"), bool):
            raise ValueError(f"{prefix}.answerable must be boolean")
        if not isinstance(case.get("language"), str) or not isinstance(case.get("category"), str):
            raise ValueError(f"{prefix} needs language and category strings")
        if not any(isinstance(case.get(key), str) and case[key].strip() for key in ("expected_answer", "answer_criteria")):
            raise ValueError(f"{prefix} needs expected_answer or answer_criteria")
        support = case.get("expected_support")
        if support is not None:
            if not isinstance(support, dict):
                raise ValueError(f"{prefix}.expected_support must be an object or null")
            for key in ("pages", "chunk_indices"):
                values = support.get(key)
                minimum = 1 if key == "pages" else 0
                if values is not None and (
                    not isinstance(values, list)
                    or any(not isinstance(value, int) or isinstance(value, bool) or value < minimum for value in values)
                ):
                    raise ValueError(f"{prefix}.expected_support.{key} must be an integer array with values >= {minimum}")
    return dataset


def load_trace(path: Path, case_ids: set[str]) -> dict[str, dict[str, Any]]:
    trace = _load_json(path, "trace")
    entries = trace.get("results")
    if not isinstance(entries, list):
        raise ValueError("Trace must contain a results array")
    indexed: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(entries):
        prefix = f"Trace result {index}"
        if not isinstance(entry, dict):
            raise ValueError(f"{prefix} must be an object")
        case_id = entry.get("case_id")
        if not isinstance(case_id, str) or case_id not in case_ids:
            raise ValueError(f"{prefix} has an unknown or invalid case_id")
        if case_id in indexed:
            raise ValueError(f"Trace contains duplicate results for {case_id}")
        selected_id = entry.get("selected_document_id")
        if not isinstance(selected_id, str) or not selected_id.strip():
            raise ValueError(f"{prefix}.selected_document_id must be a non-empty string")
        retrieval = entry.get("retrieval")
        provided = entry.get("provided_source_ids")
        citations = entry.get("citations")
        if not isinstance(retrieval, list):
            raise ValueError(f"{prefix}.retrieval must be an array")
        if provided is not None and not isinstance(provided, list):
            raise ValueError(f"{prefix}.provided_source_ids must be an array when observed")
        if citations is not None and not isinstance(citations, list):
            raise ValueError(f"{prefix}.citations must be an array when observed")
        if any(not isinstance(source_id, str) or not source_id for source_id in (provided or [])):
            raise ValueError(f"{prefix}.provided_source_ids must contain non-empty strings")
        for match_index, match in enumerate(retrieval):
            if not isinstance(match, dict) or not isinstance(match.get("document_id"), str):
                raise ValueError(f"{prefix}.retrieval[{match_index}] needs a document_id")
            for key in ("page_number", "chunk_index"):
                value = match.get(key)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ValueError(f"{prefix}.retrieval[{match_index}].{key} must be a non-negative integer")
            if "source_id" in match and not isinstance(match["source_id"], str):
                raise ValueError(f"{prefix}.retrieval[{match_index}].source_id must be a string")
            if "score" in match and match["score"] is not None and (
                not isinstance(match["score"], (int, float)) or isinstance(match["score"], bool) or not math.isfinite(match["score"])
            ):
                raise ValueError(f"{prefix}.retrieval[{match_index}].score must be a finite number")
        for citation_index, citation in enumerate(citations or []):
            if not isinstance(citation, dict) or not isinstance(citation.get("source_id"), str) or not citation["source_id"]:
                raise ValueError(f"{prefix}.citations[{citation_index}] needs a source_id")
            page = citation.get("page_number")
            if page is not None and (not isinstance(page, int) or isinstance(page, bool) or page < 1):
                raise ValueError(f"{prefix}.citations[{citation_index}].page_number must be a positive integer")
            if citation.get("document_id") is not None and not isinstance(citation["document_id"], str):
                raise ValueError(f"{prefix}.citations[{citation_index}].document_id must be a string")
        prediction = entry.get("predicted_answerable")
        if prediction is not None and not isinstance(prediction, bool):
            raise ValueError(f"{prefix}.predicted_answerable must be boolean")
        latency = entry.get("latency_ms")
        if latency is not None and (
            not isinstance(latency, (int, float)) or isinstance(latency, bool) or not math.isfinite(latency) or latency < 0
        ):
            raise ValueError(f"{prefix}.latency_ms must be a finite non-negative number")
        if "answer" in entry and not isinstance(entry["answer"], str):
            raise ValueError(f"{prefix}.answer must be a string")
        indexed[case_id] = entry
    return indexed


def _metric(numerator: int, denominator: int, *, eligible_cases: int, missing_label_cases: int, unit: str) -> dict[str, Any]:
    return {
        "value": numerator / denominator if denominator else None,
        "numerator": numerator,
        "denominator": denominator,
        "eligible_cases": eligible_cases,
        "missing_label_cases": missing_label_cases,
        "unit": unit,
    }


def evaluate(dataset: dict[str, Any], trace_by_case: dict[str, dict[str, Any]], k: int = 5) -> dict[str, Any]:
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    cases = dataset["cases"]
    hits = chunk_relevant = page_hits = 0
    chunk_labelled = page_labelled_cases = 0
    chunk_eval_cases = page_eval_cases = 0
    citation_valid = citation_count = 0
    supporting_correct = supporting_count = 0
    supporting_cases = answerability_correct = answerability_count = 0
    citation_trace_cases = 0
    supporting_page_trace_cases = 0
    answerability_labelled_cases = 0
    latencies: list[float] = []

    for case in cases:
        trace = trace_by_case.get(case["case_id"])
        support = case.get("expected_support") or {}
        expected_chunks = set(support.get("chunk_indices") or [])
        expected_pages = set(support.get("pages") or [])
        if expected_chunks:
            chunk_labelled += 1
        if expected_pages:
            page_labelled_cases += 1
        if case.get("answerable") is not None:
            answerability_labelled_cases += 1
        if trace is None:
            continue

        selected_id = trace["selected_document_id"]
        expected_document_id = case.get("document_id")
        target_id = expected_document_id if expected_document_id else selected_id
        selected_document_matches = not expected_document_id or selected_id == expected_document_id
        retrieved = [
            item for item in trace["retrieval"][:k]
            if selected_document_matches and item["document_id"] == target_id
        ]
        if expected_chunks:
            chunk_eval_cases += 1
            found_chunks = {item["chunk_index"] for item in retrieved}
            chunk_relevant += len(found_chunks & expected_chunks)
            hits += int(bool(found_chunks & expected_chunks))
        if expected_pages:
            page_eval_cases += 1
            found_pages = {item["page_number"] for item in retrieved}
            page_hits += int(bool(found_pages & expected_pages))

        provided_field = trace.get("provided_source_ids")
        citations_field = trace.get("citations")
        provided_ids = set(provided_field or [])
        citation_metadata_observed = provided_field is not None and citations_field is not None
        citation_trace_cases += int(citation_metadata_observed)
        if expected_pages and citation_metadata_observed:
            supporting_page_trace_cases += 1
        retrieval_by_source = {
            item.get("source_id"): item for item in trace["retrieval"] if item.get("source_id")
        }
        for citation in citations_field or []:
            if citation_metadata_observed:
                citation_count += 1
                citation_valid += int(citation["source_id"] in provided_ids)
            if expected_pages and citation_metadata_observed:
                supporting_count += 1
                match = retrieval_by_source.get(citation["source_id"], {})
                cited_document_id = citation.get("document_id", match.get("document_id"))
                cited_page = citation.get("page_number", match.get("page_number"))
                citation_was_supplied = citation["source_id"] in provided_ids
                supporting_correct += int(
                    selected_document_matches and citation_was_supplied and cited_document_id == target_id
                    and cited_page in expected_pages
                )
                supporting_cases += 1

        prediction = trace.get("predicted_answerable")
        if prediction is not None:
            answerability_count += 1
            answerability_correct += int(prediction == case["answerable"])
        latency = trace.get("latency_ms")
        if latency is not None:
            latencies.append(float(latency))

    case_count = len(cases)
    missing_trace_cases = case_count - len(trace_by_case)
    metrics = {
        "retrieval_hit_rate_at_k": _metric(
            hits, chunk_eval_cases, eligible_cases=chunk_eval_cases,
            missing_label_cases=case_count - chunk_labelled, unit="cases with >=1 exact relevant chunk",
        ),
        "chunk_recall_at_k": _metric(
            chunk_relevant, sum(len(set((case.get("expected_support") or {}).get("chunk_indices") or [])) for case in cases if case["case_id"] in trace_by_case),
            eligible_cases=chunk_eval_cases, missing_label_cases=case_count - chunk_labelled,
            unit="document-scoped relevant chunk indices",
        ),
        "page_hit_rate_at_k": _metric(
            page_hits, page_eval_cases, eligible_cases=page_eval_cases,
            missing_label_cases=case_count - page_labelled_cases, unit="cases with >=1 retrieved supporting page",
        ),
        "supporting_page_accuracy": _metric(
            supporting_correct, supporting_count, eligible_cases=supporting_cases,
            missing_label_cases=case_count - supporting_page_trace_cases,
            unit="citations resolving to an expected supporting page",
        ),
        "citation_id_validity": _metric(
            citation_valid, citation_count, eligible_cases=citation_trace_cases,
            missing_label_cases=case_count - citation_trace_cases, unit="citations whose source_id was supplied to the model",
        ),
        "answerability_accuracy": _metric(
            answerability_correct, answerability_count, eligible_cases=answerability_count,
            missing_label_cases=case_count - answerability_count, unit="cases with predicted_answerable labels",
        ),
        "response_latency_ms": {
            "sample_count": len(latencies),
            "mean": mean(latencies) if latencies else None,
            "median": median(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
            "unit": "milliseconds supplied by trace; no requests made",
        },
    }
    return {
        "report_version": 1,
        "dataset_name": dataset.get("name", "unnamed dataset"),
        "illustrative_labels": bool(dataset.get("illustrative", False)),
        "k": k,
        "sample_counts": {
            "dataset_cases": case_count,
            "trace_results": len(trace_by_case),
            "missing_trace_cases": missing_trace_cases,
            "chunk_relevance_labelled_cases": chunk_labelled,
            "page_relevance_labelled_cases": page_labelled_cases,
            "answerability_labelled_cases": answerability_labelled_cases,
        },
        "metrics": metrics,
        "limitations": LIMITATIONS,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Offline RAG Evaluation: {report['dataset_name']}",
        "",
        f"- Illustrative labels: **{'yes' if report['illustrative_labels'] else 'no'}**",
        f"- K: **{report['k']}**",
        f"- Samples: **{report['sample_counts']['trace_results']} / {report['sample_counts']['dataset_cases']}** traces; **{report['sample_counts']['missing_trace_cases']}** missing",
        "",
        "| Metric | Value | Numerator / denominator | Evaluated cases | Missing labels |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metric in report["metrics"].items():
        if "value" in metric:
            value = "N/A" if metric["value"] is None else f"{metric['value']:.3f}"
            lines.append(f"| {name} | {value} | {metric['numerator']} / {metric['denominator']} | {metric['eligible_cases']} | {metric['missing_label_cases']} |")
        else:
            values = metric["sample_count"]
            summary = "N/A" if not values else f"mean {metric['mean']:.1f}, median {metric['median']:.1f}, max {metric['max']:.1f}"
            lines.append(f"| {name} | {summary} | {values} supplied values | {values} | N/A |")
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--k", default=5, type=int)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--markdown-out", required=True, type=Path)
    args = parser.parse_args()
    try:
        dataset = load_dataset(args.dataset)
        trace = load_trace(args.trace, {case["case_id"] for case in dataset["cases"]})
        report = evaluate(dataset, trace, args.k)
    except ValueError as exc:
        parser.error(str(exc))
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown_out.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote offline evaluation report: {args.json_out} and {args.markdown_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
