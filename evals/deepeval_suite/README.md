# DeepEval for this project

Run commands from the project root. This package adapts the uploaded evaluation
suite to the actual hybrid retriever, Groq generator, and FastAPI application.
Existing RAGAS commands and the existing evaluation dashboard still work.

## Start here

```bash
uv pip install --python .venv/bin/python -r requirements.txt
uv run python -m evals.deepeval_suite.evaluate --validate-only
```

Validation checks dataset shape without contacting an LLM or the application.
The starter set contains five questions whose reference passages were checked
against `DATA/true_data`, and eight safety cases based on the current guard policy.
These are smoke tests, not a complete quality or security benchmark. The existing
15-question RAGAS dataset is preserved; some of its references need more evidence
before they are suitable for isolated generator testing.

The installed/tested DeepEval version is pinned to 4.2.3. No separate DeepEval
login, OpenAI key, or dashboard is required. The judge uses `JUDGE_GROQ`, falling
back to `GROQ_API_KEY` in the root `.env`. `DEEPEVAL_JUDGE_MODEL` overrides the
default `openai/gpt-oss-20b`. Sharing the application model/key is convenient but
can correlate judge and answer errors and share provider quotas. Review scores
against human judgments before relying on them for deployment decisions.

## First live check: generator only

This works before hybrid reingestion because it supplies verified passages to
the actual generator. It calls Groq for generation and for judgment.

```bash
uv run python -m evals.deepeval_suite.evaluate --level generator --limit 1 --metrics faithfulness
```

Default: one case, sequential scoring, a 10-second pause between judge calls.
One metric can make several judge calls. Change pacing in `config.json` if needed;
it does not guarantee that every provider quota will be avoided. Requests have
timeouts and one SDK retry for the judge. Errors stay visible in reports.

## Retrieve and test the complete application

Old dense-only Qdrant data cannot serve the new hybrid retriever. First populate
the new collection (leave the previous collection intact):

```bash
uv run python -m app.ingestion.processor DATA/true_data true
```

Both ingestion and the API must use `QDRANT_COLLECTION=enterprise_rag_hybrid_v1`
(the default). An explicit environment override takes precedence. Do not use
`--wipe` for this migration. Cached dense embeddings can be reused.

After ingestion succeeds, start/restart the backend in a separate terminal:

```bash
uv run uvicorn main:app --port 8000
```

Then run selected checks:

```bash
uv run python -m evals.deepeval_suite.evaluate --level retriever --limit 1
uv run python -m evals.deepeval_suite.evaluate --level pipeline --limit 1
uv run python -m evals.deepeval_suite.evaluate --level safety --limit 0 --metrics guardrail_decision
uv run python -m evals.deepeval_suite.evaluate --level all --limit 1
```

`--limit 0` means all cases; `--limit 1` on `all` runs one document case and one
safety case. `component` groups retriever and generator; `application` groups
quality, safety and operations. `EVAL_API_URL` changes the `/query` URL.

| Level | Measured behavior |
|---|---|
| retriever | Actual hybrid + FlashRank results: contextual precision and recall |
| generator | Actual generator with verified evidence: faithfulness and answer relevancy |
| pipeline | API answer and actual used evidence: context relevancy, faithfulness, answer relevancy |
| quality | Correctness, completeness and style against project references/policies |
| safety | Exact allow/block decision, scope adherence, non-leakage and toxicity |
| ops | Observed latency percentiles, reliability and available cost coverage |
| online | Reference-free scoring of a supplied trace file; no live monitoring daemon |

For DeepEval 4.2.3, **higher toxicity scores mean safer/non-toxic output**; the
threshold here is 0.95. This differs from the uploaded suite's assumption.

Cost stays null because the API does not measure total usage across all stages.
The cost gate is disabled until configured; this is explicitly recorded. Setting
`max_mean_cost_usd` enables it, and missing costs then prevent a passing result.
Judge token usage is separate, excludes unobserved retry usage, and is not the
application's cost. Tiny samples provide descriptive latency only, not a load test.

## Reports and regression

Each run prints a summary and saves a unique JSON report under `reports/`.
It includes full answers, exact evidence, sources, decisions, scores, reasons,
error coverage, local app/config metadata and judge usage. These files may contain
document content; they are ignored by Git. No reporting service upload is used.

```bash
uv run python -m evals.deepeval_suite.evaluate --level pipeline --limit 0 --output evals/deepeval_suite/reports/baseline.json
# After a deliberate application change, repeat the same workload and judge:
uv run python -m evals.deepeval_suite.evaluate --level pipeline --limit 0 --baseline evals/deepeval_suite/reports/baseline.json --output evals/deepeval_suite/reports/candidate.json
```

Comparison rejects changes to cases, judge configuration, metric selection or
evaluation code. Application changes are recorded and allowed. Judge errors or
incomplete coverage cannot pass a comparison. A baseline below the absolute
quality threshold can still be compared; the candidate must meet its own gates
to pass overall. Thresholds are initial targets to calibrate, not universal facts.
Record corpus changes separately; the report names the collection but does not
snapshot its contents. API metadata describes the local checkout, so verify that
the running backend matches it.

Exit codes: 0 = all selected checks passed; 1 = failed, errored or unscorable checks.
Argument/configuration errors stop the command. Empty grounding evidence is
unscorable for faithfulness; missing retrieval for document questions scores zero.

## Recorded trace evaluation

Supply a JSONL file, one object per request, with `trace_id`, `input`, `answer`,
`retrieval_context` (texts actually used), `route` (`rag`, `conversational`, or
`blocked`), `success` (boolean), `latency_s`, and optional measured `cost_usd`.
Failure rows need ID, route, success and latency; they remain in reliability counts.

```bash
uv run python -m evals.deepeval_suite.evaluate --level online --traces path/to/traces.jsonl --sample-rate 0.1
```

Sampling is deterministic by trace ID. All requests contribute to operations;
only sampled successful requests are judged. Document grounding applies only to
the RAG route. No sampled quality checks means no quality pass. This does not
automatically export Logfire traces or collect every production request. Judge
work happens after the application response. These tests are single-turn;
conversation-memory evaluation needs a separate multi-turn dataset.

## Files

- `project_adapter.py`: calls production components and API; isolates conversations.
- `judge.py`: Groq judge with schema validation, pacing and usage counts.
- `metrics.py`: constructs DeepEval metrics and checks guardrail decisions.
- `evaluate.py`: runs selected levels, saves reports and compares baselines.
- `config.json`: thresholds, policies, pacing and operational gates.
- `goldens.json`, `safety.json`: source-backed questions and expected guard behavior.

Local verification:

```bash
uv run python -m unittest discover -s tests -q
```

Unit tests validate adapters, reference passages, metric construction, evidence
isolation, failures, sampling and comparisons. They do not establish answer quality.
