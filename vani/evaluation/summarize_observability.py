"""Summarize VANI JSONL chat traces without accessing services or raw content."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRACE = PROJECT_ROOT / "data" / "observability" / "chat_traces.jsonl"
DEFAULT_JSON = PROJECT_ROOT / "data" / "observability" / "summary.json"
DEFAULT_MARKDOWN = PROJECT_ROOT / "data" / "observability" / "summary.md"
OUTCOMES = ("success", "fallback", "error")
RETRIEVAL_PATHS = ("vector", "first_page", "first_page_plus_vector", "none")
LATENCY_FIELDS = (
    "overall_duration_ms",
    "retrieval_duration_ms",
    "llm_duration_ms",
)
TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
EXPECTED_FIELDS = {
    "timestamp_utc", "request_id", "overall_duration_ms", "retrieval_duration_ms",
    "llm_duration_ms", "retrieval_path", "retrieved_chunk_count", "citations_returned",
    "used_fallback", "fallback_reason", "outcome", "model", "prompt_tokens",
    "completion_tokens", "total_tokens",
}


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return value


def _count(value: Any) -> int | None:
    number = _number(value)
    if isinstance(number, int):
        return number
    return None


def _distribution(values: list[int | float], missing: int, include_sum: bool = False) -> dict[str, Any]:
    sorted_values = sorted(values)
    result: dict[str, Any] = {
        "observed_count": len(values),
        "missing_count": missing,
        "average": round(statistics.fmean(values), 3) if values else None,
        "median": round(statistics.median(values), 3) if values else None,
        # Nearest-rank percentile: rank=ceil(0.95*n), with one-based ranks.
        "p95": round(sorted_values[max(0, math.ceil(0.95 * len(values)) - 1)], 3) if values else None,
    }
    if include_sum:
        result["sum"] = sum(values) if values else 0
    return result


def summarize_records(records: Iterable[dict[str, Any]], malformed_lines: int = 0,
                      non_object_lines: int = 0, total_lines: int | None = None) -> dict[str, Any]:
    rows = list(records)
    request_count = len(rows)
    if total_lines is None:
        total_lines = request_count + malformed_lines + non_object_lines

    outcomes = {key: 0 for key in OUTCOMES}
    outcomes["unknown"] = 0
    paths = {key: 0 for key in RETRIEVAL_PATHS}
    paths["unknown"] = 0
    latency_values: dict[str, list[int | float]] = {key: [] for key in LATENCY_FIELDS}
    token_values: dict[str, list[int]] = {key: [] for key in TOKEN_FIELDS}
    latency_missing = {key: 0 for key in LATENCY_FIELDS}
    token_missing = {key: 0 for key in TOKEN_FIELDS}
    fallback_observed = 0
    fallback_true = 0
    fallback_missing = 0
    citation_observed = 0
    citation_present = 0
    citation_missing = 0
    incomplete_records = 0
    reason_counts: dict[str, int] = {}

    # Only allowlisted categorical values and numeric counts enter the report.
    for row in rows:
        if EXPECTED_FIELDS - row.keys():
            incomplete_records += 1
        outcome = row.get("outcome")
        outcomes[outcome if outcome in OUTCOMES else "unknown"] += 1
        path = row.get("retrieval_path")
        paths[path if path in RETRIEVAL_PATHS else "unknown"] += 1

        for field in LATENCY_FIELDS:
            value = _number(row.get(field))
            if value is None:
                latency_missing[field] += 1
            else:
                latency_values[field].append(value)

        for field in TOKEN_FIELDS:
            value = _count(row.get(field))
            if value is None:
                token_missing[field] += 1
            else:
                token_values[field].append(value)

        fallback = row.get("used_fallback")
        if isinstance(fallback, bool):
            fallback_observed += 1
            fallback_true += int(fallback)
        else:
            fallback_missing += 1

        citations = _count(row.get("citations_returned"))
        if citations is None:
            citation_missing += 1
        else:
            citation_observed += 1
            citation_present += int(citations > 0)

        reason = row.get("fallback_reason")
        if isinstance(reason, str) and reason in {
            "missing_document_selection", "empty_retrieval", "document_isolation_failure",
            "empty_model_response", "invalid_or_missing_citations", "model_error", "other",
        }:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    latency = {
        field: _distribution(latency_values[field], latency_missing[field])
        for field in LATENCY_FIELDS
    }
    tokens = {
        field: _distribution(token_values[field], token_missing[field], include_sum=True)
        for field in TOKEN_FIELDS
    }
    return {
        "schema_version": 1,
        "counts": {
            "total_lines": total_lines,
            "valid_json_records": request_count,
            "malformed_json_lines": malformed_lines,
            "non_object_json_lines": non_object_lines,
            "incomplete_records": incomplete_records,
        },
        "requests_by_outcome": outcomes,
        "requests_by_retrieval_path": paths,
        "latency_ms": latency,
        "fallback_rate": {
            "fallback_requests": fallback_true,
            "observed_requests": fallback_observed,
            "missing_count": fallback_missing,
            "rate": round(fallback_true / fallback_observed, 6) if fallback_observed else None,
        },
        "citation_presence_rate": {
            "requests_with_one_or_more_citations": citation_present,
            "observed_requests": citation_observed,
            "missing_count": citation_missing,
            "rate": round(citation_present / citation_observed, 6) if citation_observed else None,
        },
        "token_usage": tokens,
        "fallback_reason_counts": reason_counts,
        "definitions": {
            "fallback_rate": "fallback_requests / requests with a boolean used_fallback field",
            "citation_presence_rate": "requests with citations_returned > 0 / requests with a non-negative integer citations_returned field",
            "latency_percentile": "P95 uses nearest-rank (ceil(0.95*n)); only finite, non-negative observed values are included",
            "token_usage": "Only non-negative integer token counts recorded from provider usage are included; totals are not derived",
        },
        "limitations": [
            "Citation presence is not citation validity, accuracy, or support quality.",
            "A successful request does not establish answer correctness, relevance, or faithfulness.",
            "This report summarizes operational traces only and does not evaluate semantic answer quality.",
            "Malformed JSON lines and non-object JSON values are counted but excluded from request metrics.",
        ],
    }


def read_trace_file(path: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    malformed = 0
    non_object = 0
    total = 0
    with path.open("r", encoding="utf-8", errors="replace") as trace_file:
        for line in trace_file:
            total += 1
            try:
                value = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                malformed += 1
                continue
            if not isinstance(value, dict):
                non_object += 1
                continue
            records.append(value)
    return summarize_records(records, malformed, non_object, total)


def render_markdown(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    lines = [
        "# VANI chat observability summary", "",
        f"- JSONL lines: {counts['total_lines']}",
        f"- Valid JSON object records: {counts['valid_json_records']}",
        f"- Malformed JSON lines: {counts['malformed_json_lines']}",
        f"- Non-object JSON lines: {counts['non_object_json_lines']}",
        f"- Incomplete records: {counts['incomplete_records']}", "",
        "## Requests by outcome", "",
        "| Outcome | Requests |", "|---|---:|",
    ]
    lines.extend(f"| {key} | {count} |" for key, count in summary["requests_by_outcome"].items())
    lines.extend(["", "## Requests by retrieval path", "", "| Path | Requests |", "|---|---:|"])
    lines.extend(f"| {key} | {count} |" for key, count in summary["requests_by_retrieval_path"].items())
    lines.extend(["", "## Latency (milliseconds)", "", "| Metric | Observed | Missing | Average | Median | P95 |", "|---|---:|---:|---:|---:|---:|"])
    for metric, values in summary["latency_ms"].items():
        lines.append(f"| {metric} | {values['observed_count']} | {values['missing_count']} | {values['average']} | {values['median']} | {values['p95']} |")

    fallback = summary["fallback_rate"]
    citations = summary["citation_presence_rate"]
    lines.extend([
        "", "## Rates", "",
        f"- Fallback rate: {fallback['rate']} ({fallback['fallback_requests']} / {fallback['observed_requests']} observed boolean values; missing {fallback['missing_count']}).",
        f"- Citation presence rate: {citations['rate']} ({citations['requests_with_one_or_more_citations']} / {citations['observed_requests']} observed citation counts; missing {citations['missing_count']}).",
        "", "## Provider-reported token usage", "",
        "| Token field | Observed | Missing | Sum | Average | Median | P95 |", "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for metric, values in summary["token_usage"].items():
        lines.append(f"| {metric} | {values['observed_count']} | {values['missing_count']} | {values['sum']} | {values['average']} | {values['median']} | {values['p95']} |")
    lines.extend(["", "## Fallback reasons", ""])
    if summary["fallback_reason_counts"]:
        lines.extend(f"- `{reason}`: {count}" for reason, count in sorted(summary["fallback_reason_counts"].items()))
    else:
        lines.append("No recognized fallback reason codes were recorded.")
    lines.extend(["", "## Interpretation limits", ""])
    lines.extend(f"- {item}" for item in summary["limitations"])
    lines.append("")
    return "\n".join(lines)


def _write_reports(summary: dict[str, Any], json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(summary), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_TRACE, help="Read-only JSONL trace file")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON, help="JSON report output path")
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN, help="Markdown report output path")
    args = parser.parse_args(argv)
    source = args.input.resolve()
    outputs = [args.json_out.resolve(), args.markdown_out.resolve()]
    if source in outputs or outputs[0] == outputs[1]:
        parser.error("input and report output paths must be different")
    try:
        summary = read_trace_file(args.input)
    except OSError:
        parser.error("unable to read the specified trace file")
    try:
        _write_reports(summary, args.json_out, args.markdown_out)
    except OSError:
        parser.error("unable to write one or both report files")
    print(f"Read {summary['counts']['total_lines']} trace lines; summarized {summary['counts']['valid_json_records']} records.")
    print(f"JSON report: {args.json_out}")
    print(f"Markdown report: {args.markdown_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
