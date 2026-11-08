"""Collect read-only retrieval traces from an already-running local VANI API."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from api.qa import classify_question_intent
from evaluation.evaluator import evaluate, load_dataset, render_markdown


DOCUMENT_ID = "12cc259a-5f4f-479d-a4c4-0a1583d67a51"
VECTOR_TOP_K = 10
CHAT_VECTOR_TOP_K = 5
CHAT_FIRST_PAGE_TOP_K = 6
CHAT_SOURCE_LIMIT = 10
REQUEST_TIMEOUT_SECONDS = 15


class CollectionError(ValueError):
    """Sanitized local collection failure; never includes response bodies."""

    def __init__(self, error_type: str, http_status: int | None = None) -> None:
        super().__init__(error_type)
        self.error_type = error_type
        self.http_status = http_status


def _validate_local_base_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise CollectionError("non_local_base_url")
    return base_url.rstrip("/")


def _request_json(
    base_url: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = Request(
        base_url + path,
        data=data,
        headers={"Accept": "application/json", **({"Content-Type": "application/json"} if data is not None else {})},
        method="POST" if data is not None else "GET",
    )
    try:
        with opener(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        exc.close()
        raise CollectionError("http_error", exc.code) from None
    except (URLError, TimeoutError, OSError):
        raise CollectionError("connection_error") from None
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise CollectionError("invalid_json_response") from None
    if not isinstance(parsed, dict):
        raise CollectionError("invalid_response_shape")
    return parsed


def _valid_integer(value: Any, minimum: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _safe_search_results(response: dict[str, Any], document_id: str) -> list[dict[str, Any]]:
    results = response.get("results")
    if not isinstance(results, list):
        raise CollectionError("invalid_search_results")
    safe: list[dict[str, Any]] = []
    for item in results[:VECTOR_TOP_K]:
        if not isinstance(item, dict):
            raise CollectionError("invalid_search_result")
        if item.get("document_id") != document_id:
            raise CollectionError("document_isolation_violation")
        if not _valid_integer(item.get("page_number"), 1) or not _valid_integer(item.get("chunk_index"), 0):
            raise CollectionError("invalid_search_location")
        if not isinstance(item.get("text"), str):
            raise CollectionError("invalid_search_text_field")
        record: dict[str, Any] = {
            "rank": len(safe) + 1,
            "document_id": document_id,
            "page_number": item["page_number"],
            "chunk_index": item["chunk_index"],
            "_has_text": bool(item["text"].strip()),
        }
        score = item.get("score")
        if score is not None:
            if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score):
                raise CollectionError("invalid_search_score")
            record["score"] = score
        safe.append(record)
    return safe


def _safe_first_page(response: dict[str, Any], document_id: str) -> tuple[list[dict[str, Any]], bool, int]:
    if response.get("document_id") != document_id:
        raise CollectionError("document_isolation_violation")
    matches = response.get("first_page_chunks")
    chunk_count = response.get("first_page_chunk_count")
    if (
        not isinstance(matches, list)
        or not isinstance(response.get("first_page_truncated"), bool)
        or not _valid_integer(chunk_count, 0)
        or len(matches) > chunk_count
    ):
        raise CollectionError("invalid_diagnostic_response")
    safe: list[dict[str, Any]] = []
    for item in matches:
        if not isinstance(item, dict) or item.get("document_id") != document_id:
            raise CollectionError("document_isolation_violation")
        if item.get("page_number") != 1 or not _valid_integer(item.get("chunk_index"), 0):
            raise CollectionError("invalid_diagnostic_location")
        if not isinstance(item.get("excerpt"), str):
            raise CollectionError("invalid_diagnostic_excerpt")
        safe.append({
            "document_id": document_id,
            "page_number": 1,
            "chunk_index": item["chunk_index"],
            "_has_text": bool(item["excerpt"].strip()),
        })
    safe.sort(key=lambda item: item["chunk_index"])
    return safe, response["first_page_truncated"], chunk_count


def _failure(case_id: str, error: CollectionError) -> dict[str, Any]:
    failure: dict[str, Any] = {"case_id": case_id, "error_type": error.error_type}
    if error.http_status is not None:
        failure["http_status"] = error.http_status
    return failure


def _result(case_id: str, document_id: str, path: str, retrieval: list[dict[str, Any]]) -> dict[str, Any]:
    safe_retrieval = []
    for item in retrieval:
        record = {key: item[key] for key in ("rank", "document_id", "page_number", "chunk_index", "score") if key in item}
        safe_retrieval.append(record)
    return {
        "case_id": case_id,
        "selected_document_id": document_id,
        "retrieval_path": path,
        "retrieval": safe_retrieval,
    }


def collect_baselines(
    dataset: dict[str, Any],
    *,
    base_url: str = "http://127.0.0.1:8000",
    document_id: str = DOCUMENT_ID,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    """Query only the running local API; never create a Qdrant client or invoke /chat."""
    local_url = _validate_local_base_url(base_url)
    try:
        health = _request_json(local_url, "/health", opener=opener)
    except CollectionError as exc:
        raise CollectionError(f"backend_unavailable:{exc.error_type}", exc.http_status) from None
    if health.get("status") != "ok":
        raise CollectionError("backend_health_not_ok")

    first_page: list[dict[str, Any]] | None = None
    first_page_truncated = False
    first_page_chunk_count = 0
    diagnostic_error: CollectionError | None = None
    try:
        diagnostic_response = _request_json(
            local_url,
            f"/diagnostics/retrieval/{document_id}",
            opener=opener,
        )
        first_page, first_page_truncated, first_page_chunk_count = _safe_first_page(diagnostic_response, document_id)
    except CollectionError as exc:
        diagnostic_error = exc

    vector_results: list[dict[str, Any]] = []
    vector_failures: list[dict[str, Any]] = []
    by_case_id: dict[str, list[dict[str, Any]]] = {}
    for case in dataset["cases"]:
        case_id = case["case_id"]
        if case.get("document_id") != document_id:
            vector_failures.append(_failure(case_id, CollectionError("dataset_document_id_mismatch")))
            continue
        try:
            response = _request_json(
                local_url,
                "/search",
                payload={"question": case["question"], "top_k": VECTOR_TOP_K, "document_id": document_id},
                opener=opener,
            )
            results = _safe_search_results(response, document_id)
            by_case_id[case_id] = results
            vector_results.append(_result(case_id, document_id, "vector_search_top10", results))
        except CollectionError as exc:
            vector_failures.append(_failure(case_id, exc))

    chat_results: list[dict[str, Any]] = []
    chat_failures: list[dict[str, Any]] = []
    for case in dataset["cases"]:
        case_id = case["case_id"]
        if case_id not in by_case_id:
            chat_failures.append(_failure(case_id, CollectionError("vector_results_unavailable")))
            continue

        intent = classify_question_intent(case["question"])
        vector = by_case_id[case_id]
        if intent in {"authors", "title"}:
            if first_page is None:
                error = diagnostic_error or CollectionError("first_page_candidates_unavailable")
                chat_failures.append(_failure(case_id, error))
                continue
            path = "first_page"
            candidates = first_page[:CHAT_SOURCE_LIMIT]
        elif intent == "overview":
            if first_page is None:
                error = diagnostic_error or CollectionError("first_page_candidates_unavailable")
                chat_failures.append(_failure(case_id, error))
                continue
            path = "first_page_plus_vector"
            candidates = first_page[:CHAT_FIRST_PAGE_TOP_K] + vector[:CHAT_VECTOR_TOP_K]
            deduplicated: list[dict[str, Any]] = []
            seen: set[tuple[str, int, int]] = set()
            for item in candidates:
                key = (item["document_id"], item["page_number"], item["chunk_index"])
                if key not in seen:
                    seen.add(key)
                    deduplicated.append(item)
            candidates = deduplicated
        else:
            path = "vector"
            candidates = vector[:CHAT_VECTOR_TOP_K]

        candidates = [item for item in candidates[:CHAT_SOURCE_LIMIT] if item.get("_has_text", True)]
        chat_results.append(_result(case_id, document_id, path, [
            {**item, "rank": index + 1} for index, item in enumerate(candidates)
        ]))

    vector_trace = {
        "trace_version": 1,
        "retrieval_path": "vector_search_top10",
        "source_endpoint": "POST /search",
        "document_id": document_id,
        "results": vector_results,
        "failed_cases": vector_failures,
    }
    chat_trace = {
        "trace_version": 1,
        "retrieval_path": "chat_candidate_reconstruction",
        "source_endpoints": ["POST /search", "GET /diagnostics/retrieval/{document_id}"],
        "document_id": document_id,
        "candidate_policy": "Mirrors current api/qa.py intent routing and source limits; vector lists come from POST /search, page-one candidates from the local read-only diagnostic route.",
        "first_page_diagnostics_available": first_page is not None,
        "first_page_truncated": first_page_truncated,
        "first_page_chunk_count": first_page_chunk_count,
        "results": chat_results,
        "failed_cases": chat_failures,
    }
    return {
        "vector_trace": vector_trace,
        "chat_trace": chat_trace,
        "coverage": {
            "dataset_cases": len(dataset["cases"]),
            "vector_search_succeeded": len(vector_results),
            "vector_search_failed": len(vector_failures),
            "chat_candidates_reconstructed": len(chat_results),
            "chat_candidates_unavailable": len(chat_failures),
            "first_page_diagnostics_available": first_page is not None,
            "first_page_truncated": first_page_truncated,
            "first_page_chunk_count": first_page_chunk_count,
            "chat_candidate_paths": dict(Counter(entry["retrieval_path"] for entry in chat_results)),
        },
        "diagnostic_failure": _failure("_first_page_diagnostic", diagnostic_error) if diagnostic_error else None,
    }


def _make_report(dataset: dict[str, Any], trace: dict[str, Any], path_name: str) -> tuple[dict[str, Any], str]:
    case_ids = {case["case_id"] for case in dataset["cases"]}
    trace_by_case = {entry["case_id"]: entry for entry in trace["results"]}
    report = evaluate(dataset, trace_by_case, k=VECTOR_TOP_K)
    report["retrieval_path"] = path_name
    report["retrieval_coverage"] = {
        "dataset_cases": len(case_ids),
        "successful_cases": len(trace_by_case),
        "failed_or_missing_cases": len(case_ids) - len(trace_by_case),
    }
    report["failed_cases"] = trace["failed_cases"]
    report["candidate_path_counts"] = dict(Counter(entry.get("retrieval_path", "unspecified") for entry in trace["results"]))
    report["limitations"] = [
        limitation for limitation in report["limitations"]
        if "runner does not call an LLM or connect to Qdrant" not in limitation
        and (report["illustrative_labels"] or "Illustrative example labels are not ground truth" not in limitation)
    ]
    report["limitations"].append(
        "Retrieval rows were collected from the already-running local API; collection used read-only endpoints, made no LLM request, and created no local Qdrant client."
    )
    markdown = render_markdown(report)
    markdown += f"\n## Retrieval path\n\n`{path_name}`\n\n"
    markdown += "Candidate routes per case:\n\n"
    if report["candidate_path_counts"]:
        for candidate_path, count in sorted(report["candidate_path_counts"].items()):
            markdown += f"- `{candidate_path}`: {count}\n"
    else:
        markdown += "- No successful retrieval cases.\n"
    markdown += "\n"
    markdown += (
        f"Coverage: **{len(trace_by_case)} / {len(case_ids)}** cases; "
        f"**{len(trace['failed_cases'])}** failed or unavailable.\n\n"
    )
    markdown += "## Failed or unavailable cases\n\n"
    if trace["failed_cases"]:
        for failure in trace["failed_cases"]:
            status = f", HTTP {failure['http_status']}" if "http_status" in failure else ""
            markdown += f"- `{failure['case_id']}`: `{failure['error_type']}{status}`\n"
    else:
        markdown += "None.\n"
    return report, markdown


def write_baselines(dataset: dict[str, Any], collection: dict[str, Any], output_dir: Path) -> None:
    if output_dir.exists():
        raise CollectionError("output_dir_exists")
    output_dir.mkdir(parents=True, exist_ok=False)
    traces = {}
    if collection["vector_trace"]["results"]:
        traces["vector_search"] = collection["vector_trace"]
    if collection["chat_trace"]["results"]:
        traces["chat_candidates"] = collection["chat_trace"]
    for name, trace in traces.items():
        (output_dir / f"{name}_trace.json").write_text(
            json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        report, markdown = _make_report(dataset, trace, trace["retrieval_path"])
        (output_dir / f"{name}_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (output_dir / f"{name}_report.md").write_text(markdown, encoding="utf-8")
    summary = {key: value for key, value in collection.items() if key not in {"vector_trace", "chat_trace"}}
    (output_dir / "collection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/real_dataset.json"))
    parser.add_argument("--document-id", default=DOCUMENT_ID)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/real_retrieval_baseline"))
    args = parser.parse_args()
    if args.output_dir.exists():
        print("Collection stopped safely: output_dir_exists", file=sys.stderr)
        return 2
    try:
        dataset = load_dataset(args.dataset)
        if any(case.get("document_id") != args.document_id for case in dataset["cases"]):
            raise CollectionError("dataset_document_id_mismatch")
        collection = collect_baselines(dataset, base_url=args.base_url, document_id=args.document_id)
    except (ValueError, CollectionError) as exc:
        safe = exc.error_type if isinstance(exc, CollectionError) else "invalid_dataset"
        status = f" HTTP {exc.http_status}" if isinstance(exc, CollectionError) and exc.http_status else ""
        print(f"Collection stopped safely: {safe}{status}", file=sys.stderr)
        return 2
    write_baselines(dataset, collection, args.output_dir)
    print(json.dumps(collection["coverage"], ensure_ascii=False))
    print(f"Wrote real retrieval traces and separate reports under {args.output_dir}")
    print("No chat request, LLM call, or local Qdrant client was used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
