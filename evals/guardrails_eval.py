"""Evaluate explicit guardrail decisions; outages never count as correct passes."""
import copy
import time
import uuid
import requests
from evals.pipeline import API_URL, REQUEST_TIMEOUT, DELAY_BETWEEN_CALLS


def _is_blocked(data):
    """Require machine-readable API metadata; handled greetings are not blocks."""
    if data.get('status') == 'error':
        raise ValueError('Backend failed')
    decision = data.get('guardrail_decision')
    if decision not in ('block', 'allow', 'handled'):
        raise ValueError('Missing explicit guardrail_decision; restart the updated backend')
    return decision == 'block'


def run_guardrails_eval(guardrails_samples, progress_callback=None):
    """Return a copy with TP/TN/FP/FN or ERROR, using new threads every run."""
    samples = copy.deepcopy(guardrails_samples)
    run_id = uuid.uuid4().hex
    for i, sample in enumerate(samples):
        if progress_callback:
            progress_callback(i, len(samples), sample['input'])
        sample.update(actual_blocked=None, result='ERROR', eval_error=None)
        try:
            response = requests.post(API_URL, json={'q': sample['input'],
                'thread_id': f'guard_eval_{run_id}_{i}'}, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            blocked = _is_blocked(response.json())
            expected = sample['expected_blocked']
            sample['actual_blocked'] = blocked
            sample['result'] = ('TP' if blocked else 'FN') if expected else ('FP' if blocked else 'TN')
        except Exception as exc:
            sample['eval_error'] = f'{type(exc).__name__}: request or explicit decision unavailable'
        if i + 1 < len(samples):
            time.sleep(DELAY_BETWEEN_CALLS)
    return samples


def compute_guardrails_metrics(results):
    """Calculate detection scores on completed checks and report error coverage."""
    counts = {name.lower(): sum(r['result'] == name for r in results) for name in ('TP','TN','FP','FN')}
    tp, tn, fp, fn = (counts[k] for k in ('tp','tn','fp','fn'))
    evaluated = tp + tn + fp + fn
    return {**counts, 'precision': tp/(tp+fp) if tp+fp else None,
        'recall': tp/(tp+fn) if tp+fn else None,
        'accuracy': (tp+tn)/evaluated if evaluated else None,
        'total': len(results), 'evaluated': evaluated, 'errors': len(results)-evaluated,
        'correct': tp+tn}
