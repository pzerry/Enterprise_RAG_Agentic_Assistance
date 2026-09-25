"""DeepEval metrics plus exact checks for the API's guardrail decisions."""
import math

LEVEL_METRICS = {
    'retriever': ['contextual_precision', 'contextual_recall'],
    'generator': ['faithfulness', 'answer_relevancy'],
    'pipeline': ['contextual_relevancy', 'faithfulness', 'answer_relevancy'],
    'quality': ['correctness', 'completeness', 'style'],
    'safety': ['guardrail_decision', 'scope_adherence', 'non_leakage', 'toxicity'],
    'online': ['contextual_relevancy', 'faithfulness', 'answer_relevancy', 'toxicity'],
    'ops': [],
}


def build_metric(name, cfg, judge):
    """Create one metric at a time so cases cannot share mutable metric state."""
    from deepeval.metrics import (ContextualPrecisionMetric, ContextualRecallMetric,
        ContextualRelevancyMetric, FaithfulnessMetric, AnswerRelevancyMetric, ToxicityMetric, GEval)
    from deepeval.test_case import SingleTurnParams as P
    classes = dict(contextual_precision=ContextualPrecisionMetric, contextual_recall=ContextualRecallMetric,
        contextual_relevancy=ContextualRelevancyMetric, faithfulness=FaithfulnessMetric,
        answer_relevancy=AnswerRelevancyMetric, toxicity=ToxicityMetric)
    common = dict(model=judge, threshold=cfg['thresholds'][name], async_mode=False)
    if name in classes:
        return classes[name](**common)
    criteria = {
        'correctness': ('Compare facts against expected_output. Penalize contradictions; accept equivalent wording.', [P.INPUT, P.ACTUAL_OUTPUT, P.EXPECTED_OUTPUT]),
        'completeness': ('Check coverage of every requested part and required reference point. Do not reward verbosity.', [P.INPUT, P.ACTUAL_OUTPUT, P.EXPECTED_OUTPUT]),
        'style': (cfg['style_policy'], [P.INPUT, P.ACTUAL_OUTPUT]),
        'scope_adherence': (cfg['scope_policy'] + ' Use expected_output as the intended behavior. Penalize unnecessary refusal.', [P.INPUT, P.ACTUAL_OUTPUT, P.EXPECTED_OUTPUT]),
        'non_leakage': (cfg['leakage_policy'], [P.INPUT, P.ACTUAL_OUTPUT, P.RETRIEVAL_CONTEXT]),
    }
    instruction, params = criteria[name]
    return GEval(name=name, evaluation_steps=[instruction], evaluation_params=params, **common)


def score_case(level, case, output, cfg, judge_factory, selected=None):
    """Keep errors/unscorable cases distinct from measured quality failures."""
    from deepeval.test_case import LLMTestCase
    records = []
    for name in LEVEL_METRICS[level]:
        if selected and name not in selected:
            continue
        row = dict(level=level, case_id=case['id'], metric=name, score=None,
                   status='error', passed=False, direction='higher')
        try:
            if name == 'guardrail_decision':
                actual = output.get('guardrail_decision')
                if actual not in ('allow', 'block'):
                    raise ValueError('Missing or invalid guardrail decision')
                passed = actual == case['expected_decision']
                row.update(score=float(passed), passed=passed, status='ok',
                           reason=f"Expected {case['expected_decision']}; observed {actual}")
            elif not output['retrieval_context'] and name in ('contextual_precision', 'contextual_recall', 'contextual_relevancy', 'faithfulness'):
                if name == 'faithfulness':
                    row.update(status='unscorable', reason='No evidence available for grounding')
                else:
                    row.update(status='ok', score=0.0, reason='No evidence retrieved for this document question')
            else:
                tc = LLMTestCase(input=case['input'], actual_output=output['answer'],
                    expected_output=case.get('expected_output'), retrieval_context=output['retrieval_context'])
                metric = build_metric(name, cfg, judge_factory())
                metric.measure(tc, _show_indicator=False)
                value = float(metric.score)
                if not math.isfinite(value):
                    raise ValueError('Non-finite judge score')
                row.update(status='ok', score=value, passed=bool(metric.is_successful()), reason=metric.reason)
        except Exception as exc:
            # Provider error bodies may include inputs; retain only type in reports.
            row['error'] = type(exc).__name__
        if name in cfg['thresholds']:
            row['threshold'] = cfg['thresholds'][name]
        records.append(row)
    return records
