# Offline RAG Evaluation: vani-illustrative-baseline

- Illustrative labels: **yes**
- K: **5**
- Samples: **8 / 8** traces; **0** missing

| Metric | Value | Numerator / denominator | Evaluated cases | Missing labels |
|---|---:|---:|---:|---:|
| retrieval_hit_rate_at_k | 1.000 | 6 / 6 | 6 | 2 |
| chunk_recall_at_k | 1.000 | 8 / 8 | 6 | 2 |
| page_hit_rate_at_k | 1.000 | 6 / 6 | 6 | 2 |
| supporting_page_accuracy | 1.000 | 6 / 6 | 6 | 2 |
| citation_id_validity | 1.000 | 6 / 6 | 8 | 0 |
| answerability_accuracy | 1.000 | 8 / 8 | 8 | 0 |
| response_latency_ms | mean 390.0, median 380.0, max 520.0 | 8 supplied values | 8 | N/A |

## Limitations

- This evaluator does not score semantic answer correctness, completeness, or faithfulness.
- Answerability accuracy uses the explicitly supplied predicted_answerable trace label; it does not infer answerability from answer text.
- Chunk relevance uses document-scoped chunk_index labels; a page match is reported separately and is never counted as a chunk match.
- Illustrative example labels are not ground truth for arbitrary PDFs. Replace them with human-verified labels before interpreting scores.
- All metrics use supplied files only; the runner does not call an LLM or connect to Qdrant.
