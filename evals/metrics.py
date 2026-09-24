"""RAGAS scoring of complete captured answers/evidence, with explicit errors.

Scores describe completed checks, not API availability. Failed checks retain an
error row rather than becoming zeroes or disappearing. No reference contexts
are substituted for retrieved evidence. The judge is separate from the answer
model call, but may use the same Groq key unless JUDGE_GROQ is configured.
"""
import asyncio
import os
import pandas as pd

METRIC_NAMES = ['faithfulness', 'answer_relevancy', 'context_precision',
                'context_recall', 'answer_correctness', 'tool_correctness']
JUDGE_MODEL = os.getenv('EVAL_JUDGE_MODEL', 'openai/gpt-oss-20b')
COOLDOWN_SECONDS = float(os.getenv('EVAL_SCORE_DELAY', '10'))


def _prep_samples(dataset):
    """Keep every sample so collection failures remain visible in score tables."""
    return dataset['rag_samples']


def _build_judge():
    """Create an OpenAI-compatible Groq judge; return its client for cleanup."""
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    key = os.getenv('JUDGE_GROQ') or os.getenv('GROQ_API_KEY')
    if not key:
        raise ValueError('Set JUDGE_GROQ or GROQ_API_KEY in the project .env')
    client = AsyncOpenAI(api_key=key, base_url='https://api.groq.com/openai/v1',
                         timeout=60, max_retries=0)
    options = {'reasoning_effort': 'low'} if JUDGE_MODEL.startswith('openai/gpt-oss-') else {}
    llm = llm_factory(JUDGE_MODEL, provider='openai', client=client,
                      max_tokens=4096, temperature=0, **options)
    return llm, client


async def run_all_metrics(golden_dataset, status_cb=None, selected=None):
    """Return one DataFrame per selected metric, including failed/skipped rows.

    Run sequentially with configurable pauses; pauses are not a quota guarantee.
    HTTP/provider failures are not quality scores. Start with one sample and
    faithfulness/tool correctness to avoid downloading a local embedding model.
    """
    selected = METRIC_NAMES if selected is None else selected
    if not selected or any(name not in METRIC_NAMES for name in selected):
        raise ValueError('Select at least one known metric')
    samples = _prep_samples(golden_dataset)
    if not samples:
        raise ValueError('No evaluation samples selected')
    results, client, judge, embeddings = {}, None, None, None
    model_calls = 0
    try:
        for name in selected:
            rows = []
            for sample in samples:
                row = {'id': sample.get('id'), 'question': sample['question'],
                       name: float('nan'), 'status': 'error', 'error': None}
                if sample.get('eval_status') != 'ok':
                    row['error'] = 'Collection failed or not run; recollect this sample'
                    rows.append(row)
                    continue
                if name == 'tool_correctness':
                    actual = set(sample.get('actual_tools_called') or [])
                    expected = set(sample.get('expected_tools') or [])
                    if 'unknown' in actual:
                        row['error'] = 'API route could not be inferred'
                    else:
                        union = actual | expected
                        row.update({name: len(actual & expected)/len(union) if union else 1.0,
                                    'status': 'ok'})
                    rows.append(row)
                    continue
                contexts = sample.get('actual_contexts', [])
                if name in ('faithfulness','context_precision','context_recall') and not contexts:
                    if name == 'faithfulness':
                        row.update(status='unscorable', error='No retrieved evidence for grounding check')
                    else:
                        row.update({name: 0.0, 'status': 'ok'})
                    rows.append(row)
                    continue
                if status_cb:
                    status_cb(f"Scoring {name}: sample {sample.get('id')}")
                try:
                    if model_calls:
                        await asyncio.sleep(COOLDOWN_SECONDS)
                    model_calls += 1
                    if judge is None:
                        judge, client = _build_judge()
                    from ragas.metrics.collections import (Faithfulness, AnswerRelevancy,
                        ContextPrecision, ContextRecall, AnswerCorrectness)
                    classes = dict(faithfulness=Faithfulness, answer_relevancy=AnswerRelevancy,
                        context_precision=ContextPrecision, context_recall=ContextRecall,
                        answer_correctness=AnswerCorrectness)
                    kwargs = {'llm': judge}
                    if name in ('answer_relevancy','answer_correctness'):
                        if embeddings is None:
                            from ragas.embeddings import HuggingFaceEmbeddings
                            embeddings = HuggingFaceEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2', use_api=False)
                        kwargs['embeddings'] = embeddings
                    inputs = {'user_input': sample['question']}
                    if name in ('faithfulness','answer_relevancy','answer_correctness'):
                        inputs['response'] = sample['actual_response']
                    if name in ('faithfulness','context_precision','context_recall'):
                        inputs['retrieved_contexts'] = contexts
                    if name in ('context_precision','context_recall','answer_correctness'):
                        inputs['reference'] = sample['reference']
                    score = await classes[name](**kwargs).ascore(**inputs)
                    import math
                    value = float(score.value)
                    if not math.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError('Judge returned an invalid score')
                    row.update({name: value, 'status': 'ok'})
                except Exception as exc:
                    # Do not expose raw prompts, keys or provider response bodies.
                    row['error'] = f'{type(exc).__name__}: judge/dependency failure; check model access and quota'
                rows.append(row)
            results[name] = pd.DataFrame(rows)
    finally:
        if client is not None:
            await client.close()
    return results
