# RAG document ingestion

For the new terminal-based DeepEval component, application and regression suite,
see [evals/deepeval_suite/README.md](evals/deepeval_suite/README.md).
The existing RAGAS evaluation dashboard remains available.

This project extracts text, splits it into bounded chunks, creates Gemini
embeddings, and stores vectors in Qdrant. The current changes address oversized
chunks, rate limits, duplicate records on reruns, and misleading success logs.

## Run

Use the existing virtual environment and configure `GEMINI_API_KEY`,
`QDRANT_CLUSTER_ENDPOINT`, and `QDRANT_API_KEY` in `.env`.

```bash
uv pip install --python .venv/bin/python 'google-genai>=1.75.0' fonttools
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m app.ingestion.processor DATA/noisy_5 noisy
```

The same ingestion command can be rerun after an interruption. Successful
unchanged documents are skipped after checking their points still exist.
Successful chunk embeddings are cached even if a later chunk fails.
A nonzero exit code means a file failed or the run stopped early.

The default collection is **enterprise_rag_hybrid_v1**. Existing dense collections
remain untouched. Reingest into this collection before restarting the API.
Existing cached Gemini embeddings are reused for unchanged chunks; BM25 vectors
are computed locally. Do not use `--wipe` for this migration. If you explicitly
set `QDRANT_COLLECTION` in your environment, change it to the hybrid collection
for both ingestion and the API.

## Follow the code

1. `app/config.py` reads credentials and configurable limits.
2. `run_universal_ingestion()` prepares Qdrant, ensures keyword indexes on
   `document_id` and `version` for strict-mode cleanup, and groups input folders.
3. `process_file()` calculates the document identity and version, and checks
   whether that version has already been indexed.
4. `chunk_text()` keeps paragraphs together where possible and splits oversized
   paragraphs. Every chunk is at most 1,500 **characters** by default.
5. `embed_texts()` embeds chunks in order, using one input per request to avoid
   multi-input aggregation. `_embed()` adds matching retrieval prefixes for
   documents/queries and first checks the persistent vector cache.
6. `_reserve()` accounts for every uncached request and retry before sending it.
   When the rolling minute budget is full, it waits. `_embed()` honors provider
   retry delays and applies bounded backoff for transient failures.
7. `process_file()` checks vector counts/dimensions, writes stable Qdrant IDs,
   removes old versions after successful writes, and saves an indexed manifest.

All functions in the chunker, embedding service, and processor have docstrings.
For example, inspect one from Python:

```python
from app.services.retrieval.embeddings import embed_texts
help(embed_texts)
```

## Settings

These optional `.env` values have the following defaults:

| Variable | Default | Meaning |
| --- | --- | --- |
| `EMBEDDING_MODEL` | `gemini-embedding-2-preview` | Preserve the explicitly selected model; no automatic fallback |
| `EMBEDDING_DIM` | `3072` | Expected vector width |
| `EMBEDDING_RPM` | `80` | Local requests/minute budget, below the observed 100 quota |
| `EMBEDDING_TPM` | `24000` | Estimated tokens/minute budget, below the observed 30,000 quota |
| `EMBEDDING_RPD` | `1000` | Local request reservations per rolling 24 hours |
| `EMBEDDING_MAX_ATTEMPTS` | `6` | Maximum attempts for an uncached embedding |
| `EMBEDDING_STATE` | `.ingestion/embeddings.sqlite3` | Persistent quota history and vector cache |
| `CHUNK_SIZE` | `1500` | Hard character bound for extracted chunks |
| `QDRANT_COLLECTION` | `enterprise_rag_hybrid_v1` | Named dense + sparse collection |

Token costs use **UTF-8 byte length plus overhead**, a conservative estimate,
not Gemini's exact tokenizer. This intentionally favors staying under quota
rather than maximizing throughput. A maximum estimated input cost also rejects
oversized input before sending it. Requests are single-input batches; smaller
batches do not by themselves solve TPM without the pacing step.

## Recovery and limits

- Quota reservations survive restarts and are shared by local processes using
  the same SQLite file and model name. The tracker cannot see usage before it
  was installed, or traffic from other apps/machines. Provider limits remain
  authoritative; 429 responses are still possible.
- Daily reservations use a conservative rolling 24-hour window. Google's daily
  reset is midnight Pacific time; the local tracker may wait longer. Recognized
  provider daily quota failures stop immediately. Repeated unknown 429s stop
  after bounded retries. Resume later using the same command.
- Cache keys include model, vector dimension, and prefixed input. A startup
  probe no longer spends quota, and an outage cannot silently switch models.
- Manifests under `processed_data/<source>/` are named by document ID and record
  `pending`, `failed`, or `indexed`. Old filename-based JSON is not treated as
  evidence of successful indexing.
- Stable document identity uses absolute path and source label. Copying a file
  to another path treats it as a separate document; this is not global content
  deduplication. Deleted source files are not automatically removed from Qdrant.
- Updating a document is resumable, not an atomic collection-wide transaction.
  Old and new versions may coexist during a partial update. Run one ingestion
  per source document at a time, and retry failed runs before serving the new
  collection.
- The `--wipe` flag explicitly deletes the configured collection. It is not
  needed for ordinary reruns.

## Validation

The unittest suite uses temporary SQLite state, mocked provider responses, and
in-memory Qdrant. It verifies chunk bounds/content, quota waiting, persistent
reservations, server retry delays, cache reuse, daily quota stops, deterministic
IDs after partial writes, changed-document cleanup, missing remote points, and
vector mismatch rejection. It does not prove live provider availability or the
accuracy of all PDF extraction.

Provider references: [Gemini rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)
and [Gemini embeddings](https://ai.google.dev/gemini-api/docs/embeddings).

## Document guardrails

`app/Guardrails/colang_rules.py` defines the policies and NeMo flows.
`app/Guardrails/rails.py` exposes three primary functions:

- `initialize_rails()` — call once at application startup; requires `GROQ_API_KEY`.
  Optional `GUARDRAIL_MODEL` defaults to `openai/gpt-oss-20b`.
- `guard_input(question)` — call before retrieval. Continue only when
  `result.decision == 'allow'`; otherwise return `result.response`.
- `guard_output(question, answer, passages)` — call before showing a generated
  answer, using the exact ordered evidence passed to the answer model. Each
  passage is `{'source': 'filename.pdf', 'text': 'retrieved passage'}`. Instruct
  the answer model to cite these passages as `[1]`, `[2]`, etc. Show the answer
  only if the check allows it.

The compatibility function `guard(question)` returns the previous `(stop,
response)` tuple. Use the explicit result in new code to distinguish a policy
block from a greeting or temporary service failure. Exact greetings are static
responses and do not use the guard model. Other input checks and output checks
each use one guard-model call; provider retries are disabled and timeouts return
an error. These Groq calls have limits separate from Gemini embedding quotas.

Quick live input check (sends the question to Groq):

```bash
.venv/bin/python - <<'PY'
import logfire
from app.Guardrails.rails import initialize_rails, guard_input
logfire.configure(send_to_logfire=False)
initialize_rails()
print(guard_input('What compiler optimizations are in my documents?'))
PY
```

This is a synchronous API for scripts. Do not call it directly inside an async
request handler's event loop; run it in a worker thread if integrating FastAPI.
Do not reinitialize it per request.

Scope is questions potentially supported by uploaded documents, not a fixed IT
whitelist. The guard cannot determine document coverage before retrieval. Empty
retrieval produces an explicit no-evidence result. Output checks validate numeric
citation IDs and ask a model to assess grounding/safety; citations alone do not
prove support. The initial implementation has no separate retrieved-document
scanner, PII detector, user authorization, or claim of perfect injection defense.
Document instructions must also be treated as untrusted in the answer prompt.

These functions are ready for the future question-answering entry point;
ingestion is unchanged. There is not yet an end-to-end guarded answer generator
in this repository. Tests exercise real NeMo flows with simulated model decisions,
including exact evidence delivery, one-call behavior, malformed labels, failures,
and citation validation. They do not measure live attack-detection accuracy.

## Evaluation suite

Start the backend and the evaluation dashboard in separate terminals:

```bash
uv run uvicorn main:app --reload --port 8000
uv run streamlit run evals/app.py --server.port 8502
```

Open the dashboard at http://localhost:8502. Start with one question, click
**Collect RAG answers**, inspect the complete answer and evidence, then click
**Score selected metrics**. Faithfulness and tool correctness are selected by
default. Other metrics can be enabled after this smoke test. The 15 reference
questions cover `DATA/true_data` Kubernetes documents. If these are not yet
indexed, run `uv run python -m app.ingestion.processor DATA/true_data true`.

The guardrail test button runs six separate cases. The backend must be restarted
after updates: evaluation requires its explicit `guardrail_decision` field and
does not guess from response wording. It uses the current async NeMo input-check
interface. Blocks, passes, and execution failures are distinct outcomes.

Optional environment variables:

- `EVAL_API_URL`: defaults to `http://localhost:8000/query`.
- `EVAL_JUDGE_MODEL`: defaults to `openai/gpt-oss-20b` on Groq.
- `JUDGE_GROQ`: optional judge key; otherwise uses `GROQ_API_KEY`. Separate keys
  do not necessarily have separate organization/project quotas.
- `EVAL_REQUEST_DELAY`: seconds between collected questions, default 2.
- `EVAL_SCORE_DELAY`: seconds between model-based sample scores, default 10.

Pauses are not quota guarantees. Rate limits and timeouts produce error rows;
no score is invented for a failed judge request. Means cover scored rows only.
Download JSON/CSV results from the dashboard for inspection. Each new collection
run gets unique conversation IDs, preventing memory from earlier runs affecting
results. Full answers and actual contexts are retained; reference contexts are
never substituted for missing retrieval. Grounding is unscorable without evidence;
context precision/recall are zero for empty retrieval in this answerable dataset.

`langchain-community==0.3.31` and `ragas==0.4.3` are pinned because the newer
community package removed a VertexAI import still used unconditionally by this
RAGAS version. No Google/VertexAI credentials are needed for these Groq judges.
Relevancy and correctness use a local MiniLM embedding model downloaded once.

Run the regression tests with:

```bash
uv run python -m unittest discover -s tests -v
```

A synthetic scoring smoke test verifies metric integration, not your RAG's
quality. Only scores on real collected answers measure this application. The
six guardrail cases are a smoke test, not proof of comprehensive attack defense.

## Hybrid retrieval and migration

The technical route now uses Gemini dense search (up to 40 hits) and local
`Qdrant/bm25` sparse search (up to 40 hits). Qdrant applies corpus IDF and
Reciprocal Rank Fusion to select up to 15 candidates. FlashRank TinyBERT reranks
them and keeps up to 5. A stopword-only sparse query uses the dense branch.
RRF scores are ranking scores, not cosine similarity or confidence percentages.

- `app/services/retrieval/sparse_embeddings.py`: document/query BM25 encoders.
- `app/ingestion/processor.py`: named vectors, versioned reingestion, existing recovery.
- `app/services/retrieval/qdrant_service.py`: schema validation and hybrid fusion.
- `app/services/retrieval/ranking_service.py`: preserves metadata; marks reranker failures explicitly.
- `app/agents/nodes/retriever.py`: assigns numbered citation IDs after reranking.
- `app/agents/nodes/responder.py`: prompts for evidence-based citations; returns only
  passages actually included in the prompt. Conversation responses clear sources.
- Both chat UIs show source names, passage text and separate retrieval/rerank scores.
- `evals/pipeline.py`: retains source objects in `actual_sources` and extracts their
  text into `actual_contexts` for the existing evaluation metrics.

Run from the project directory:

```bash
uv pip install --python .venv/bin/python -r requirements.txt
uv run python -m app.ingestion.processor DATA/true_data true
# Optional: index the noisy test documents too.
uv run python -m app.ingestion.processor DATA/noisy_5 noisy
# After ingestion succeeds, restart the API and chat UI in separate terminals.
uv run uvicorn main:app --reload --port 8000
uv run streamlit run ui/app.py
```

FastEmbed may download assets on its first run. Reingestion must finish before
hybrid retrieval has all your documents. Cloud migration is not performed by
unit tests. Run the same evaluation questions after migration to measure whether
hybrid retrieval improves your results. Prompted citations are not an implemented
output grounding guardrail: claim-to-source verification remains separate work.
