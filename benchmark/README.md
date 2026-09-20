# Evaluation protocol

This folder turns the existing V1 checks into a repeatable benchmark. It separates deterministic checks from human judgement rather than pretending that hallucination can be measured from a keyword match alone.

## Build the dataset

1. Copy `questions.sample.json` to `questions.json`.
2. Create 30–50 cases for one or more uploaded papers. Balance A1/A2/B/C categories.
3. For paper-dependent cases, add the expected evidence section and one or more discriminative terms that must appear in retrieved evidence.
4. Add a `thread_id` per case, or pass one shared `--thread-id` when running the benchmark.

Do not commit private paper text, raw answers, API keys, or local production thread IDs.

## Run deterministic checks

```bash
python benchmark/run_benchmark.py \
  --cases benchmark/questions.json \
  --thread-id YOUR_LOCAL_THREAD_ID \
  --output benchmark/reports/v1.json
```

The report records routing accuracy, retrieval hit rate, evidence size, and route/retrieval latency. It does not generate a final answer, so it does not add answer-generation cost to the benchmark.

## Review answer quality

For a fixed set of answers generated through the UI, fill `review_template.csv` with:

- `citation_correct`: cited section/page/excerpt supports the claim.
- `grounded`: paper facts are supported by retrieved evidence.
- `hallucination`: answer contains an unsupported paper fact.

Then summarize it:

```bash
python benchmark/summarize_human_review.py benchmark/review.csv
```

## Token and latency claims

Use `metrics_report.py` after collecting real traffic to report p50/p95 latency, evidence characters, and provider usage tokens when returned. For before/after token comparisons, run the same benchmark on two tagged commits and label any character-based calculation as an estimate; do not present a simulated value as provider billing data.
