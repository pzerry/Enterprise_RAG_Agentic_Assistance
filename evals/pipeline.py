"""Collect complete API answers and actual evidence using isolated conversations."""
import copy
import json
import os
import time
import uuid
from pathlib import Path
import requests

API_URL = os.getenv('EVAL_API_URL', 'http://localhost:8000/query')
DELAY_BETWEEN_CALLS = float(os.getenv('EVAL_REQUEST_DELAY', '2'))
REQUEST_TIMEOUT = 120


def detect_tool(steps):
    """Infer the route from existing API plan labels, not actual tool traces."""
    text = ' '.join(steps).lower()
    if 'guardrails fired' in text:
        return 'guardrails'
    if any(s in text for s in ('intent: technical', 'search term:', 'context retrieved')):
        return 'retrieve_documents'
    if 'conversational' in text or 'memory' in text:
        return 'direct_answer'
    return 'unknown'


def run_pipeline(golden_dataset, progress_callback=None):
    """Return a copy with full answers, actual contexts, status, and isolated IDs.

    Errors are recorded separately. Never substitute reference evidence for
    failed or empty retrieval. The caller can select a subset before calling.
    """
    dataset = copy.deepcopy(golden_dataset)
    run_id = uuid.uuid4().hex
    samples = dataset['rag_samples']
    for i, sample in enumerate(samples):
        sample.update(actual_response='', actual_contexts=[], actual_sources=[], actual_tools_called=[],
                      eval_status='error', eval_error=None)
        if progress_callback:
            progress_callback(i, len(samples), sample['question'], 'calling')
        try:
            response = requests.post(API_URL, json={'q': sample['question'],
                'thread_id': f'eval_{run_id}_{i}'}, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            if data.get('status') == 'error':
                raise ValueError('Backend returned status=error')
            answer, contexts = data.get('answer'), data.get('sources', [])
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError('Backend returned an empty or invalid answer')
            if not isinstance(contexts, list):
                raise ValueError('Backend returned invalid sources')
            # Keep metadata for inspection; RAGAS consumes passage text only.
            texts = [c.get('content') if isinstance(c, dict) else c for c in contexts]
            if any(not isinstance(text, str) for text in texts):
                raise ValueError('Backend returned invalid source content')
            sample.update(actual_response=answer, actual_contexts=texts, actual_sources=contexts,
                actual_tools_called=[detect_tool(data.get('thought_process') or [])], eval_status='ok')
        except Exception as exc:
            sample['eval_error'] = f'{type(exc).__name__}: request failed or backend response invalid'
        if progress_callback:
            progress_callback(i, len(samples), sample['question'], sample['eval_status'], sample['actual_response'])
        if i + 1 < len(samples):
            time.sleep(DELAY_BETWEEN_CALLS)
    return dataset


def save_results(dataset, path):
    """Save complete collected results for later inspection/scoring."""
    Path(path).write_text(json.dumps(dataset, indent=2), encoding='utf-8')


def load_golden_dataset():
    """Load the curated reference dataset beside this module."""
    return json.loads(Path(__file__).with_name('golden_dataset.json').read_text())
