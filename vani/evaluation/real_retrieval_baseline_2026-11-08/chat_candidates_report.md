# Offline RAG Evaluation: deep-sea-paper-vanilla-rag-human-review-v1

- Illustrative labels: **no**
- K: **10**
- Samples: **11 / 14** traces; **3** missing

| Metric | Value | Numerator / denominator | Evaluated cases | Missing labels |
|---|---:|---:|---:|---:|
| retrieval_hit_rate_at_k | N/A | 0 / 0 | 0 | 14 |
| chunk_recall_at_k | N/A | 0 / 0 | 0 | 14 |
| page_hit_rate_at_k | 0.700 | 7 / 10 | 10 | 2 |
| supporting_page_accuracy | N/A | 0 / 0 | 0 | 14 |
| citation_id_validity | N/A | 0 / 0 | 0 | 14 |
| answerability_accuracy | N/A | 0 / 0 | 0 | 14 |
| response_latency_ms | N/A | 0 supplied values | 0 | N/A |

## Limitations

- This evaluator does not score semantic answer correctness, completeness, or faithfulness.
- Answerability accuracy uses the explicitly supplied predicted_answerable trace label; it does not infer answerability from answer text.
- Chunk relevance uses document-scoped chunk_index labels; a page match is reported separately and is never counted as a chunk match.
- Retrieval rows were collected from the already-running local API; collection used read-only endpoints, made no LLM request, and created no local Qdrant client.

## Retrieval path

`chat_candidate_reconstruction`

Candidate routes per case:

- `vector`: 11

Coverage: **11 / 14** cases; **3** failed or unavailable.

## Failed or unavailable cases

- `deep-sea-authors-th`: `http_error, HTTP 404`
- `deep-sea-topic-th`: `http_error, HTTP 404`
- `deep-sea-author-degrees-unanswerable-th`: `http_error, HTTP 404`
