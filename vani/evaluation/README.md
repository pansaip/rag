# Offline RAG evaluation

This runner is separate from FastAPI `/chat`. It reads a manually labeled dataset and a recorded or mocked trace, computes deterministic file-only metrics, and writes JSON and Markdown reports. It has no imports from the production retrieval stack and makes no network or Qdrant calls.

## Dataset format

See `example_dataset.json`. The top-level object has `dataset_version`, `name`, an `illustrative` flag, and `cases`. Each case includes:

- `case_id`, `question`, `language`, `category`, and boolean `answerable`.
- Exactly one of `document_id` (canonical ID for a real evaluation case) or `document_fixture` (a named fixture for synthetic/mock cases).
- Either `expected_answer` or `answer_criteria`; criteria are human-readable and are not automatically graded.
- Optional `expected_support` with `pages` and/or document-scoped `chunk_indices`. A chunk index must match exactly; matching a page is reported separately and never counts as matching a chunk.
- Set `expected_support` to `null` or omit labels when relevance is unknown. Do not label guessed relevance as ground truth.

## Trace format

See `example_trace.json` for a clearly marked mock trace. Each `results` entry has a matching `case_id`, `selected_document_id`, ordered `retrieval` rows (`document_id`, `page_number`, `chunk_index`, optional `source_id` and `score`), `provided_source_ids`, returned `citations`, optional `predicted_answerable`, optional `answer`, and optional `latency_ms`.

`provided_source_ids` must be the source IDs actually supplied to the model. Citations are objects with a `source_id` and may include `document_id` and `page_number`; absent location fields are resolved from the retrieval row carrying that source ID. The evaluator never interprets answer prose as evidence of correctness. `predicted_answerable` is an explicit recorded label, not inferred from a fallback phrase. Latency is only summarized from supplied trace values.

## Metrics and interpretation

- `retrieval_hit_rate_at_k`: fraction of chunk-labeled cases with at least one exact expected chunk index in the top K, scoped to the selected document.
- `chunk_recall_at_k`: matching relevant chunk indices divided by labeled relevant chunk indices. This is distinct from hit rate, which counts cases.
- `page_hit_rate_at_k`: separately reports whether a supporting page appeared; page matches do not contribute to chunk metrics.
- `citation_id_validity`: citations whose IDs occur in that trace's `provided_source_ids`.
- `supporting_page_accuracy`: citations resolving to one of the expected pages in the selected document, only for cases with page labels.
- `answerability_accuracy`: recorded `predicted_answerable` compared with the dataset's `answerable` label.
- `response_latency_ms`: mean, median, and max of supplied latency values only.

Every ratio reports its numerator, denominator, eligible case count, and missing-label count. A zero denominator yields `null`/`N/A`. Metrics do not establish semantic answer correctness, completeness, or faithfulness. Those require a separate method such as human assessment or a validated rubric. Example labels and the example trace are synthetic and are not valid evidence about arbitrary PDFs or production performance.

## Add a manually labeled case

1. Copy a case in `example_dataset.json` and give it a unique `case_id`.
2. Set its actual `document_id` or a controlled `document_fixture`, and record the exact question, language, category, and answerability label.
3. Have a human verify expected answer criteria and supporting page/chunk indices against the source. Omit unknown relevance labels rather than guessing.
4. Record the corresponding trace result from an existing run or a mock. Include the selected document, ranked retrieval rows, exact source IDs passed to the model, returned citation IDs/locations, answerability prediction, and latency if available.
5. Keep real traces free of credentials and unrelated document text. The evaluator does not need full passages or prompts.

## Run

From the VANI project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m evaluation.evaluator `
  --dataset evaluation/example_dataset.json `
  --trace evaluation/example_trace.json `
  --k 5 `
  --json-out evaluation/report.json `
  --markdown-out evaluation/report.md
```

The supplied trace is only to demonstrate the runner; replace it with a real recorded or explicitly mocked trace for any comparison. Keep dataset and trace versions with the baseline so later Vanilla/Hybrid runs use identical questions and labels.

## Collect a real, read-only local retrieval baseline

`collect_retrieval.py` talks only to a loopback VANI API that is already running. It does not start a server or instantiate a Qdrant client. It sends each dataset question to `POST /search` with `top_k=10` and the exact document ID, then retains only ranked page/chunk metadata and real scores. It discards returned passage text. If the local `/diagnostics/retrieval/{document_id}` route is enabled, it also reconstructs the `/chat` candidate paths from the same search endpoint and page-one diagnostic metadata; those rows are labeled separately and no `/chat` request is made. Missing diagnostics or query failures remain visible as unavailable cases.

Run from the project root, choosing an output directory that does not already exist (the script refuses to overwrite it):

```powershell
.\.venv\Scripts\python.exe -m evaluation.collect_retrieval `
  --dataset evaluation/real_dataset.json `
  --document-id 12cc259a-5f4f-479d-a4c4-0a1583d67a51 `
  --base-url http://127.0.0.1:8000 `
  --output-dir evaluation/real_retrieval_baseline_run_2
```

If the diagnostic endpoint returns 404, first-page `/chat` paths are not collected. The route is local-only and opt-in via `VANI_ENABLE_RETRIEVAL_DIAGNOSTICS=1`. Do not start a second API process while the current process holds the Qdrant lock. To enable the route later, stop the current API process cleanly, set that variable in its launch terminal, and start exactly one local API process without an `.env` file:

```powershell
# Stop the existing API process with Ctrl+C and wait for it to exit first.
$env:VANI_ENABLE_RETRIEVAL_DIAGNOSTICS = "1"
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Then run the collector against that process using a fresh output directory. This collector intentionally records no chat answer, citations, or answerability prediction, so those report metrics remain N/A.
