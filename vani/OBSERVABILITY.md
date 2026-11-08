# Local chat observability

Request-level JSONL traces are disabled by default. They are intended for local development and contain operational metadata only; they do not contain questions, answers, passages, prompts, filenames, document IDs, credentials, or exception messages.

## Enable and disable in PowerShell

Run these commands in the PowerShell session that starts the VANI API:

```powershell
$env:VANI_ENABLE_OBSERVABILITY = "1"
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload
```

Stop the API and disable tracing in that shell with:

```powershell
Remove-Item Env:VANI_ENABLE_OBSERVABILITY -ErrorAction SilentlyContinue
```

Only the exact value `1` enables writes. The trace file is `data/observability/chat_traces.jsonl`; the directory is created when the first enabled request is recorded. The file is append-only and has no automatic rotation or retention policy. Review and remove old traces according to local data-retention needs.

## Fields

- `timestamp_utc`, `request_id`: trace timestamp and random per-request identifier.
- `overall_duration_ms`: elapsed time inside the `/chat` handler.
- `retrieval_duration_ms`, `llm_duration_ms`: measured operation durations; `null` when that operation did not run.
- `retrieval_path`: `vector`, `first_page`, `first_page_plus_vector`, or `none`.
- `retrieved_chunk_count`, `citations_returned`: counts only.
- `used_fallback`, `fallback_reason`, `outcome`: fallback/error classification without exception details.
- `model`: configured model ID if it matches a conservative safe-character allowlist; otherwise `null`.
- `prompt_tokens`, `completion_tokens`, `total_tokens`: provider-reported values only; absent values are `null`.

Token labels map Responses API `input_tokens`/`output_tokens` to prompt/completion. Compatible `prompt_tokens`/`completion_tokens` fields are accepted when supplied. The total is never calculated locally.

## Inspect recent records safely

This prints only selected operational fields and does not expose trace contents beyond them:

```powershell
$tracePath = "data/observability/chat_traces.jsonl"
if (Test-Path $tracePath) {
    Get-Content $tracePath -Tail 20 | ForEach-Object {
        $_ | ConvertFrom-Json | Select-Object timestamp_utc, request_id, overall_duration_ms, retrieval_duration_ms, llm_duration_ms, retrieval_path, retrieved_chunk_count, citations_returned, used_fallback, fallback_reason, outcome, model, prompt_tokens, completion_tokens, total_tokens
    } | Format-Table -AutoSize
}
```

## Generate an offline summary

The standard-library summary tool reads the trace file only and writes aggregate reports. It does not contact the API, Groq, or Qdrant. By default it reads `data/observability/chat_traces.jsonl` and writes `data/observability/summary.json` and `data/observability/summary.md`:

```powershell
.\.venv\Scripts\python.exe evaluation\summarize_observability.py
```

To choose paths explicitly:

```powershell
.\.venv\Scripts\python.exe evaluation\summarize_observability.py `
  --input data\observability\chat_traces.jsonl `
  --json-out data\observability\summary.json `
  --markdown-out data\observability\summary.md
```

The tool refuses to use the input trace as an output path. Empty files produce zero records and null rates/distributions. Malformed JSON and non-object JSON lines are counted and excluded. Latency and token statistics include only observed finite non-negative values; every distribution reports observed and missing counts. Fallback and citation-presence rates use only records with valid corresponding fields as their denominators. Citation presence does not measure citation accuracy, and a successful request does not establish answer correctness or faithfulness. Reports contain aggregate values only and omit model names, request IDs, and all request/document content.

## Limitations

Tracing is best-effort; filesystem failures are silently ignored so they cannot fail chat requests. A process-local lock prevents interleaving among request threads in one API process, but does not coordinate multiple worker processes or separate API instances writing to the same file. Use one local API process for this JSONL sink. File permissions, backups, and retention remain the operator's responsibility. Existing local diagnostic logs are separate and controlled by their own flag.
