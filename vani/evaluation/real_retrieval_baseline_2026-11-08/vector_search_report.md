# Offline RAG Evaluation: deep-sea-paper-vanilla-rag-human-review-v1

- Illustrative labels: **no**
- K: **10**
- Samples: **14 / 14** traces; **0** missing

| Metric | Value | Numerator / denominator | Evaluated cases | Missing labels |
|---|---:|---:|---:|---:|
| retrieval_hit_rate_at_k | N/A | 0 / 0 | 0 | 14 |
| chunk_recall_at_k | N/A | 0 / 0 | 0 | 14 |
| page_hit_rate_at_k | 0.667 | 8 / 12 | 12 | 2 |
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

`vector_search_top10`

Candidate routes per case:

- `vector_search_top10`: 14

Coverage: **14 / 14** cases; **0** failed or unavailable.

## Failed or unavailable cases

None.
