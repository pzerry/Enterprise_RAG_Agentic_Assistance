"""Streamlit evaluation dashboard: collect, inspect, then score a small subset."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / '.env')
import logfire
logfire.configure(token=os.getenv('LOGFIRE_TOKEN') or None, data_dir=ROOT / '.logfire',
                  send_to_logfire='if-token-present', service_name='evals')

import asyncio
import copy
import json
import pandas as pd
import streamlit as st
from evals.pipeline import run_pipeline, load_golden_dataset, API_URL
from evals.guardrails_eval import run_guardrails_eval, compute_guardrails_metrics
from evals.metrics import run_all_metrics, METRIC_NAMES, JUDGE_MODEL

st.set_page_config(page_title='RAG Evaluation', page_icon='🧪', layout='wide')
st.title('RAG Evaluation')
st.caption('Collect actual answers and evidence before scoring. Start with one question.')
golden = load_golden_dataset()
for key in ('enriched', 'guard_results', 'scores'):
    if key not in st.session_state:
        st.session_state[key] = None

with st.expander('Reference dataset', expanded=False):
    st.dataframe(pd.DataFrame(golden['rag_samples'])[['id','domain','question','reference']])
    st.dataframe(pd.DataFrame(golden['guardrails_samples']))
    st.info('These questions target DATA/true_data Kubernetes documents. Index those documents before evaluating them. The five noisy PDFs alone do not cover these references.')

count = st.number_input('Number of RAG questions', 1, len(golden['rag_samples']), 1)
subset = copy.deepcopy(golden)
subset['rag_samples'] = subset['rag_samples'][:count]
st.caption(f'Backend: {API_URL}. Start it with: uv run uvicorn main:app --reload')

if st.button('1. Collect RAG answers'):
    st.session_state.scores = None
    progress = st.progress(0.0)
    def update(i, total, question, stage, response=''):
        progress.progress((i+1)/total if stage != 'calling' else i/total, text=f'{stage}: {question}')
    st.session_state.enriched = run_pipeline(subset, update)

if st.session_state.enriched:
    samples = st.session_state.enriched['rag_samples']
    success = sum(s['eval_status'] == 'ok' for s in samples)
    st.write(f'Collection: {success}/{len(samples)} completed; {len(samples)-success} errors.')
    st.dataframe(pd.DataFrame(samples)[['id','question','eval_status','eval_error','actual_tools_called']])
    with st.expander('Full answers and actual retrieved evidence'):
        st.json(st.session_state.enriched)
    st.download_button('Download collected results', json.dumps(st.session_state.enriched, indent=2),
                       'eval_results.json', 'application/json')

if st.button('Run guardrail tests (6 cases)'):
    with st.spinner('Checking explicit backend guardrail decisions...'):
        st.session_state.guard_results = run_guardrails_eval(golden['guardrails_samples'])
if st.session_state.guard_results is not None:
    st.dataframe(pd.DataFrame(st.session_state.guard_results))
    st.json(compute_guardrails_metrics(st.session_state.guard_results))
    st.caption('Scores exclude ERROR cases. Null means no observations for that ratio. A greeting handled by a rail is not a safety block. Six cases are a smoke test, not a security benchmark.')

st.subheader('2. Score collected answers')
st.write(f'Judge model: {JUDGE_MODEL}')
key_name = 'JUDGE_GROQ' if os.getenv('JUDGE_GROQ') else 'GROQ_API_KEY'
st.caption(f'Judge key: {key_name}. A different key does not guarantee a separate provider quota.')
st.caption('Complete answers and evidence are used. Sequential calls have configurable pauses; provider rate limits may still cause errors. Relevancy/correctness require a local embedding-model download on first use.')
selected = st.multiselect('Metrics', METRIC_NAMES, default=['faithfulness','tool_correctness'])
if st.button('Score selected metrics', disabled=not st.session_state.enriched or not selected):
    slot = st.empty()
    try:
        st.session_state.scores = asyncio.run(run_all_metrics(st.session_state.enriched, slot.info, selected))
    except Exception as exc:
        st.error(f'Evaluation could not start: {type(exc).__name__}. Check dependencies and key configuration.')
if st.session_state.scores:
    for name, df in st.session_state.scores.items():
        completed = df[df['status'] == 'ok']
        average = completed[name].mean()
        st.write(f'{name}: {len(completed)}/{len(df)} scored; mean = {average:.3f}' if len(completed) else f'{name}: no valid scores')
        st.dataframe(df, hide_index=True)
        st.download_button(f'Download {name}', df.to_csv(index=False), f'{name}.csv', 'text/csv')
    st.caption('Means cover only scored rows. Inspect error counts before comparing runs; do not interpret incomplete evaluations as overall quality.')
