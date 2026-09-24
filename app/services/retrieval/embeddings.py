"""Explicit Gemini model, persistent conservative quotas and resumable embeddings.

UTF-8 bytes + overhead are a conservative token estimate, not an exact tokenizer.
All local processes using this state file share budgets. Other applications do not.
Daily reservations use a conservative rolling 24h window, not Google's reset time.
"""
import hashlib
import json
import math
import random
import sqlite3
import time
from pathlib import Path

import logfire
from google import genai
from google.genai import types
from app.config import settings

_client = None


class QuotaExhausted(RuntimeError):
    """Stop ingestion; retrying more files would consume the same exhausted quota."""


def _db():
    """Open the shared SQLite file and create its tables when first needed.

    ``usage`` records attempted API calls, including retries. ``cache`` stores
    successful vectors so interrupted ingestion can reuse work. Callers must
    close the returned connection. Neither table stores the API key.
    """
    path = Path(settings.EMBEDDING_STATE)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.execute('CREATE TABLE IF NOT EXISTS usage (model TEXT, ts REAL, tokens INTEGER)')
    db.execute('CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, vector TEXT)')
    return db


def token_budget(text):
    """Estimate input token cost conservatively from UTF-8 bytes plus overhead.

    This deliberately overestimates typical English text. It is not Gemini's
    exact token count; the extra 128 units allow for request formatting.
    """
    return len(text.encode('utf-8')) + 128


def _reserve(tokens):
    """Reserve one request and its estimated tokens before contacting Gemini.

    A SQLite write transaction makes reservations safe across local processes
    sharing the same state file. Wait until both minute budgets allow the call.
    A 61-second window adds a small buffer; daily accounting uses a conservative
    rolling 24 hours. Reservations survive restarts and failed API calls.

    Args:
        tokens: Estimated cost of this request, including formatting overhead.

    Raises:
        ValueError: A budget is invalid or one input cannot fit the TPM budget.
        QuotaExhausted: The local daily request budget has been consumed.

    Usage from other machines/apps is unknown, so provider 429s remain possible.
    """
    if min(settings.EMBEDDING_RPM, settings.EMBEDDING_TPM, settings.EMBEDDING_RPD) <= 0:
        raise ValueError('Embedding budgets must be positive')
    if tokens > settings.EMBEDDING_TPM:
        raise ValueError('Input exceeds configured token budget; split it before embedding')
    while True:
        db = _db()
        try:
            db.execute('BEGIN IMMEDIATE')
            now = time.time()
            db.execute('DELETE FROM usage WHERE ts <= ?', (now - 86400,))
            rows = db.execute('SELECT ts,tokens FROM usage WHERE model=? ORDER BY ts',
                              (settings.EMBEDDING_MODEL,)).fetchall()
            if len(rows) >= settings.EMBEDDING_RPD:
                raise QuotaExhausted('Local daily request budget exhausted; resume later.')
            minute = [(ts, n) for ts, n in rows if ts > now - 61]
            if len(minute) < settings.EMBEDDING_RPM and sum(n for _, n in minute) + tokens <= settings.EMBEDDING_TPM:
                db.execute('INSERT INTO usage VALUES (?,?,?)', (settings.EMBEDDING_MODEL, now, tokens))
                db.commit()
                return
            delay = max(0.1, minute[0][0] + 61 - now)
            db.commit()
        finally:
            db.close()
        logfire.info('Embedding quota pacing: waiting {seconds}s', seconds=round(delay, 1))
        time.sleep(delay)


def _error_details(exc):
    """Extract structured quota/retry details from a Google SDK exception.

    Return an empty list when the provider supplies no structured details.
    Keeping these fields avoids relying on a truncated console error message.
    """
    data = getattr(exc, 'response_json', None)
    if isinstance(data, dict):
        return data.get('error', data).get('details', [])
    return []


def _retry_delay(exc):
    """Return the largest numeric retry delay supplied by the provider.

    Read Google RetryInfo's ``retryDelay`` and a numeric HTTP Retry-After header.
    Unrecognized values are ignored; the caller also applies bounded backoff.
    """
    delay = 0.0
    for detail in _error_details(exc):
        raw = detail.get('retryDelay')
        if raw:
            try:
                delay = max(delay, float(str(raw).rstrip('s')))
            except ValueError:
                pass
    response = getattr(exc, 'response', None)
    headers = getattr(response, 'headers', {}) or {}
    try:
        delay = max(delay, float(headers.get('retry-after', 0)))
    except (TypeError, ValueError):
        pass
    return delay


def get_embedding_dim():
    """Return the configured vector width without making a probe API request.

    Ingestion uses this value to create or validate its Qdrant collection.
    The model never changes automatically when an API request fails.
    """
    return settings.EMBEDDING_DIM


def _embed(text, query=False):
    """Embed one document chunk or query, reusing a cached vector if available.

    Add the matching Embedding 2 retrieval prefix, check input size, then reserve
    quota before every attempt. SDK retries are disabled so every attempted API
    call passes through our limiter. Cache only a validated successful vector.

    Args:
        text: Nonempty document chunk or search query.
        query: Select the search-query prefix instead of the document prefix.

    Returns:
        A list of finite floats with exactly EMBEDDING_DIM values.

    Raises:
        ValueError: Input is empty/too large or the returned vector is invalid.
        QuotaExhausted: Daily quota is exhausted, rate limits persist, or the
            provider requests a delay too long for this ingestion run.
        Exception: Nonretryable provider errors or exhausted transient retries.

    The cache identity includes model, dimension, and prefixed text, preventing
    accidental reuse of document vectors as query vectors or across models.
    """
    global _client
    if not text.strip():
        raise ValueError('Cannot embed empty text')
    # Embedding 2 uses explicit retrieval prefixes, one input per request.
    prepared = f'task: search result | query: {text}' if query else f'title: none | text: {text}'
    cost = token_budget(prepared)
    if cost > 7000:
        raise ValueError('Embedding input too large; split the document or shorten the query')
    key = hashlib.sha256(json.dumps([settings.EMBEDDING_MODEL, settings.EMBEDDING_DIM, prepared]).encode()).hexdigest()
    db = _db()
    try:
        row = db.execute('SELECT vector FROM cache WHERE key=?', (key,)).fetchone()
    finally:
        db.close()
    if row:
        return json.loads(row[0])
    if _client is None:
        _client = genai.Client(api_key=settings.GEMINI_API_KEY, http_options=types.HttpOptions(
            timeout=60000, retry_options=types.HttpRetryOptions(attempts=1)))
    if settings.EMBEDDING_MAX_ATTEMPTS < 1:
        raise ValueError('EMBEDDING_MAX_ATTEMPTS must be positive')
    for attempt in range(settings.EMBEDDING_MAX_ATTEMPTS):
        _reserve(cost)
        try:
            result = _client.models.embed_content(model=settings.EMBEDDING_MODEL, contents=prepared,
                # auto_truncate is Vertex-only; Gemini rejects even False.
                # Input size is checked locally above before reserving quota.
                config=types.EmbedContentConfig(output_dimensionality=settings.EMBEDDING_DIM))
            if not result.embeddings or len(result.embeddings) != 1:
                raise ValueError('Expected exactly one embedding for one chunk')
            vector = list(result.embeddings[0].values or [])
            if len(vector) != settings.EMBEDDING_DIM or not all(math.isfinite(v) for v in vector):
                raise ValueError('Invalid embedding dimension or non-finite values')
            db = _db()
            try:
                with db:
                    db.execute('INSERT OR REPLACE INTO cache VALUES (?,?)', (key, json.dumps(vector)))
            finally:
                db.close()
            return vector
        except Exception as exc:
            code = getattr(exc, 'code', None)
            details = _error_details(exc)
            # Keep structured provider quota diagnostics without logging API keys or input text.
            logfire.error('Embedding request failed: code={code}, details={details}', code=code, details=details)
            if code == 429:
                description = json.dumps(details).lower()
                if any(s in description for s in ('perday', 'per_day', 'daily')):
                    raise QuotaExhausted(f'Provider daily quota exhausted: {details}') from exc
            if code not in (429, 500, 502, 503, 504):
                raise
            if attempt + 1 == settings.EMBEDDING_MAX_ATTEMPTS:
                if code == 429:
                    raise QuotaExhausted(f'Rate limit persisted after bounded retries: {details}') from exc
                raise
            delay = max(_retry_delay(exc), min(60, 10 * 2 ** attempt)) + random.uniform(0, 1)
            if delay > 300:
                raise QuotaExhausted(f'Provider asks to wait {delay:.0f}s; resume later.') from exc
            logfire.warning('Retrying embedding in {seconds}s (attempt {attempt})', seconds=round(delay, 1), attempt=attempt + 2)
            time.sleep(delay)


def embed_query(query: str) -> list[float]:
    """Embed a search query using the same model and quota tracker as ingestion.

    Returns one vector; rejects oversized queries rather than silently truncating
    them. Queries share document ingestion's local request and token budgets.
    """
    return _embed(query, query=True)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return one vector per input chunk, preserving the input order.

    Each chunk is an individually paced request. Successful chunks are cached,
    so retrying a document after a failure reuses its completed embeddings.
    Return an empty list for no inputs; propagate failures to the processor.
    """
    # Single-input batches avoid aggregation and make each API request count explicit.
    return [_embed(text) for text in texts]
