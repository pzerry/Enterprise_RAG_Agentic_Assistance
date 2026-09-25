"""Run component, API, regression, and recorded-trace evaluations locally."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from datetime import datetime, timezone
from importlib.metadata import version
from dotenv import load_dotenv
from . import project_adapter
from .metrics import LEVEL_METRICS, score_case

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_cases(cases, safety=False):
    """Reject duplicate IDs and missing references before any provider call."""
    if not cases:
        raise ValueError('Dataset is empty')
    seen = set()
    for case in cases:
        for field in ('id', 'input', 'expected_output'):
            if not isinstance(case.get(field), str) or not case[field].strip():
                raise ValueError(f'Missing {field}')
        if case['id'] in seen:
            raise ValueError('Duplicate case ID')
        seen.add(case['id'])
        if safety:
            if case.get('expected_decision') not in ('allow', 'block'):
                raise ValueError('Expected decision must be allow or block')
        else:
            validate_texts(case.get('gold_context'))
            if not case['gold_context']:
                raise ValueError('Reference context is empty')


def validate_texts(texts):
    if not isinstance(texts, list) or any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError('Context must be a list of nonempty strings')


def capture(call, case):
    """Preserve execution failures and their latency; never score an error answer."""
    start = time.perf_counter()
    record = {'case_id': case['id'], 'input': case['input'], 'success': False}
    try:
        output = call()
        if not isinstance(output.get('answer'), str) or not output['answer'].strip():
            raise ValueError('Invalid answer')
        validate_texts(output.get('retrieval_context'))
        record.update(output, success=True)
    except Exception as exc:
        record['error'] = type(exc).__name__
    record['latency_s'] = time.perf_counter() - start
    return record


def quantile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def operations(requests, cfg):
    """Report workload latency/reliability; incomplete cost coverage stays unknown."""
    if not requests:
        return {'status': 'unscorable', 'passed': False, 'requests': 0}
    costs = [r['cost_usd'] for r in requests if isinstance(r.get('cost_usd'), (int, float))
             and not isinstance(r['cost_usd'], bool) and math.isfinite(r['cost_usd']) and r['cost_usd'] >= 0]
    result = {'requests': len(requests), 'reliability': sum(r['success'] for r in requests)/len(requests),
        'cost_coverage': len(costs)/len(requests),
        'mean_cost_usd': statistics.mean(costs) if len(costs) == len(requests) else None}
    for name, q in [('p50', .5), ('p95', .95), ('p99', .99)]:
        result[f'latency_{name}_s'] = quantile([r['latency_s'] for r in requests], q)
    gates = cfg['ops']
    cost_gate = gates['max_mean_cost_usd']
    result['cost_status'] = 'unavailable' if result['mean_cost_usd'] is None else 'measured'
    result['cost_gate_enabled'] = cost_gate is not None
    result['passed'] = (result['reliability'] >= gates['min_reliability']
        and result['latency_p95_s'] <= gates['max_p95_latency_s']
        and (cost_gate is None or (result['mean_cost_usd'] is not None and result['mean_cost_usd'] <= cost_gate)))
    result['status'] = 'ok'
    return result


def summarize(records):
    """Expose error coverage alongside means; incomplete groups cannot pass."""
    groups = {}
    for r in records:
        groups.setdefault(r['level'] + '.' + r['metric'], []).append(r)
    summary = {}
    for name, rows in groups.items():
        scored = [r['score'] for r in rows if r['status'] == 'ok']
        summary[name] = {'mean': statistics.mean(scored) if scored else None,
            'scored': len(scored), 'total': len(rows), 'passed': all(r['passed'] for r in rows),
            'direction': rows[0]['direction']}
    return summary


def compare(baseline, current, cfg):
    """Compare fixed workloads and judges; permit application code changes."""
    for key in ('dataset_hash', 'config_hash', 'levels', 'selected_metrics', 'deepeval_version', 'suite_hash', 'trace_batch_hash', 'sample_rate'):
        if baseline['metadata'].get(key) != current['metadata'].get(key):
            raise ValueError(f'Incompatible baseline: {key}')
    if baseline['summary'].keys() != current['summary'].keys():
        raise ValueError('Baseline metric coverage differs')
    comparisons = []
    for name, row in current['summary'].items():
        old = baseline['summary'][name]
        complete = all(r['scored'] == r['total'] and r['mean'] is not None for r in (old, row))
        delta = row['mean'] - old['mean'] if complete else None
        bad = delta is None or (delta > cfg['regression']['max_score_drop'] if row['direction'] == 'lower'
                               else delta < -cfg['regression']['max_score_drop'])
        comparisons.append({'metric': name, 'delta': delta, 'regressed': bad})
    if 'operations' in baseline or 'operations' in current:
        old, new = baseline.get('operations', {}), current.get('operations', {})
        for key in ('reliability', 'latency_p95_s', 'mean_cost_usd'):
            before, after = old.get(key), new.get(key)
            if key == 'mean_cost_usd' and before is None and after is None:
                comparisons.append({'metric': 'ops.cost', 'status': 'unavailable', 'regressed': False})
                continue
            bad = before is None or after is None
            if not bad:
                bad = (after < before - cfg['regression']['max_reliability_drop'] if key == 'reliability'
                       else after > before * (1 + cfg['regression']['max_latency_relative_increase']))
            comparisons.append({'metric': 'ops.' + key, 'regressed': bad})
    return comparisons


def run_suite(levels, goldens, safety, cfg, judge_factory, selected=None):
    """Reuse the same API result across pipeline, quality and operational checks."""
    report = {'records': [], 'requests': []}
    cache = {}
    for level in levels:
        for case in safety if level == 'safety' else goldens:
            print(f'{level}: {case["id"]}', flush=True)
            key = ('safety' if level == 'safety' else 'golden', case['id'])
            if level in ('retriever', 'generator'):
                call = (lambda: project_adapter.retrieve(case['input'])) if level == 'retriever' else (
                    lambda: project_adapter.generate(case['input'], case['gold_context']))
                output = capture(call, case)
                output['level'] = level
                report['requests'].append(output)
            else:
                if key not in cache:
                    cache[key] = capture(lambda: project_adapter.run(case['input']), case)
                    cache[key]['level'] = 'safety' if level == 'safety' else 'application'
                    report['requests'].append(cache[key])
                    time.sleep(cfg['request_delay_s'])
                output = cache[key]
            if not output['success']:
                report['records'].append(dict(level=level, case_id=case['id'], metric='execution',
                    status='error', score=None, passed=False, direction='higher', error=output['error']))
            elif level != 'ops':
                report['records'] += score_case(level, case, output, cfg, judge_factory, selected)
    if 'ops' in levels:
        report['operations'] = operations([v for (kind, _), v in cache.items() if kind == 'golden'], cfg)
    return report


def run_traces(path, rate, cfg, judge_factory, selected=None):
    """Evaluate a recorded JSONL batch; sampling does not repeat application calls.

    Only traces marked route=rag receive document-grounding metrics. Operations
    include every trace, even failures and traces excluded by sampling.
    """
    report = {'records': [], 'requests': []}
    seen = set()
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        trace = json.loads(line)
        tid = trace['trace_id']
        if not isinstance(tid, str) or not tid or tid in seen:
            raise ValueError('Invalid or duplicate trace ID')
        seen.add(tid)
        if not isinstance(trace['success'], bool) or isinstance(trace['latency_s'], bool) or not isinstance(trace['latency_s'], (int, float)) or not math.isfinite(trace['latency_s']) or trace['latency_s'] < 0:
            raise ValueError('Invalid trace status/latency')
        if trace.get('route') not in ('rag', 'conversational', 'blocked'):
            raise ValueError('Trace route must be rag, conversational, or blocked')
        report['requests'].append(trace)
        sampled = int(hashlib.sha256(tid.encode()).hexdigest(), 16)/2**256 < rate
        if not trace['success'] or not sampled:
            continue
        validate_texts(trace.get('retrieval_context'))
        if not isinstance(trace.get('answer'), str) or not trace['answer'].strip() or not isinstance(trace.get('input'), str) or not trace['input'].strip():
            raise ValueError('Trace input/answer missing')
        names = selected or LEVEL_METRICS['online']
        if trace['route'] != 'rag':
            names = [n for n in names if n in ('answer_relevancy', 'toxicity')]
        if names:
            report['records'] += score_case('online', {'id': tid, 'input': trace['input']}, trace, cfg, judge_factory, names)
    report['operations'] = operations(report['requests'], cfg)
    return report


def main(argv=None):
    """Validate without APIs, or run a deliberately small selection by default."""
    load_dotenv(PROJECT / '.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--level', choices=['all', 'component', 'application'] + list(LEVEL_METRICS), default='pipeline')
    parser.add_argument('--limit', type=int, default=1, help='Cases per dataset; 0 means all')
    parser.add_argument('--metrics', nargs='+', choices=sorted({n for v in LEVEL_METRICS.values() for n in v}))
    parser.add_argument('--goldens', type=Path, default=ROOT / 'goldens.json')
    parser.add_argument('--safety', type=Path, default=ROOT / 'safety.json')
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--traces', type=Path)
    parser.add_argument('--sample-rate', type=float, default=.1)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args(argv)
    if args.limit < 0 or not 0 <= args.sample_rate <= 1:
        parser.error('Require limit >= 0 and sample-rate between 0 and 1')
    cfg = read_json(args.config)
    cfg['judge_model'] = os.getenv('DEEPEVAL_JUDGE_MODEL', cfg['judge_model'])
    goldens, safety = read_json(args.goldens), read_json(args.safety)
    validate_cases(goldens)
    validate_cases(safety, True)
    if args.validate_only:
        print(f'Validated {len(goldens)} document cases and {len(safety)} safety cases. No APIs called.')
        return 0
    levels = {'all': ['retriever', 'generator', 'pipeline', 'quality', 'safety', 'ops'],
              'component': ['retriever', 'generator'], 'application': ['quality', 'safety', 'ops']}.get(args.level, [args.level])
    if args.metrics and (not set(args.metrics) <= {n for level in levels for n in LEVEL_METRICS[level]}
                         or any(level != 'ops' and not set(args.metrics).intersection(LEVEL_METRICS[level]) for level in levels)):
        parser.error('Selected metrics must cover each requested level and belong to those levels')
    if args.limit:
        goldens, safety = goldens[:args.limit], safety[:args.limit]
    judge = None
    def get_judge():
        nonlocal judge
        if judge is None:
            from .judge import GroqJudge
            judge = GroqJudge(cfg['judge_model'], cfg['judge_delay_s'])
        return judge
    import logfire
    logfire.configure(send_to_logfire=False, console=False)
    try:
        if args.level == 'online':
            if not args.traces:
                parser.error('--traces is required for online mode')
            report = run_traces(args.traces, args.sample_rate, cfg, get_judge, args.metrics)
        else:
            report = run_suite(levels, goldens, safety, cfg, get_judge, args.metrics)
        from app.config import settings
        app_files = sorted((PROJECT / 'app').rglob('*.py')) + [PROJECT / 'main.py']
        report['metadata'] = dict(created_at=datetime.now(timezone.utc).isoformat(),
            dataset_hash=fingerprint([goldens, safety]), config_hash=fingerprint(cfg),
            levels=levels, selected_metrics=args.metrics, deepeval_version=version('deepeval'),
            suite_hash=fingerprint({p.name: p.read_text() for p in sorted(ROOT.glob('*.py'))}),
            app_hash=fingerprint({str(p.relative_to(PROJECT)): p.read_text() for p in app_files}),
            judge_model=cfg['judge_model'], app_model=settings.GROQ_MODEL,
            collection=settings.QDRANT_COLLECTION, embedding_model=settings.EMBEDDING_MODEL,
            embedding_dim=settings.EMBEDDING_DIM, api_url=os.getenv('EVAL_API_URL', 'http://localhost:8000/query'),
            note='Application metadata describes local code/config; verify a remote API uses the same build.')
        if args.traces:
            report['metadata'].update(trace_batch_hash=hashlib.sha256(args.traces.read_bytes()).hexdigest(), sample_rate=args.sample_rate)
        report['summary'] = summarize(report['records'])
        report['passed'] = (all(r['passed'] for r in report['records'])
            and report.get('operations', {}).get('passed', True)
            and bool(report['records'] or (levels == ['ops'])))
        report['judge_usage'] = judge.usage if judge else {'calls': 0, 'cost_usd': None}
        if args.baseline:
            try:
                report['regression'] = compare(read_json(args.baseline), report, cfg)
                report['passed'] &= not any(r['regressed'] for r in report['regression'])
            except ValueError as exc:
                report.update(passed=False, comparison_error=str(exc))
        output = args.output or ROOT / 'reports' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f') + '.json')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        print(json.dumps({'passed': report['passed'], 'summary': report['summary'], 'operations': report.get('operations'), 'report': str(output)}, indent=2))
        return 0 if report['passed'] else 1
    finally:
        if judge:
            judge.close()


if __name__ == '__main__':
    raise SystemExit(main())
