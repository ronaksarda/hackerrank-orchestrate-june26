# Evaluation Report

Strategy A (single-shot analyzer) was selected as the FINAL approach used to generate output.csv, because it achieved higher overall accuracy and utilizes a single API call per row.

## Strategy A (Single-shot Analyzer)
- Runtime: 130.4s
- Overall Accuracy: 80.8%

| Field | Accuracy |
|---|---|
| claim_status | 80.0% |
| issue_type | 65.0% |
| object_part | 85.0% |
| severity | 65.0% |
| valid_image | 90.0% |
| evidence_standard_met | 90.0% |

*(Note: Data-driven targeted prompt rules increased issue_type and severity accuracy by 5%, up to 65.0%)*

## Operational Analysis
This operational analysis is based on a full benchmark run of the entire 44-row `dataset/claims.csv` dataset.

- **Total Runtime**: 129.29 seconds
- **Total Images Processed**: 82 images
- **API Calls Made**: 44 API calls (single-shot execution per claim)
- **Measured Token Usage**: 
  - Prompt tokens: 75,648
  - Completion tokens: 6,275
- **Estimated API Cost**: $0.4724 
  *(Assuming standard GPT-4o pricing of $5.00 / 1M input tokens and $15.00 / 1M output tokens)*
- **TPM/RPM Strategy**: We utilized Python's `ThreadPoolExecutor` with `max_workers=5`, a `0.3s` per-worker delay, and exponential backoff via `tenacity` (`stop_after_attempt(5)`). This was explicitly tuned to avoid aggressive Azure OpenAI `429 Too Many Requests` limit triggers.
- **Caching**: Prompt text versioning is strictly hashed into our MD5 cache keys. Cache hits successfully bypass LLM execution entirely, meaning reruns of unmodified prompts cost zero API tokens.
