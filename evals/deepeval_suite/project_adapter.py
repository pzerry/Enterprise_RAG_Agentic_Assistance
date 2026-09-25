"""Connect evaluation to production code; references never enter the full API."""
import os
import uuid
import requests


def retrieve(question: str) -> dict:
    """Run the real hybrid + FlashRank node, preserving its source metadata."""
    from app.agents.nodes.retriever import retrieve_node
    state = retrieve_node({'current_query': question, 'plan': []})
    return {'answer': '[retriever-only]', 'sources': state['documents'],
            'retrieval_context': [d['content'] for d in state['documents']]}


def generate(question: str, context: list[str]) -> dict:
    """Isolate the real generator with reference evidence, without retrieval.

    Return the evidence actually used, since the generator has a context limit.
    The runner rejects truncation for component tests to preserve isolation.
    """
    from app.agents.nodes.responder import generate_node
    docs = [{'id': i, 'source': 'Reviewed reference passage', 'content': text}
            for i, text in enumerate(context, 1)]
    state = generate_node({'current_query': question, 'plan': [], 'documents': docs,
                          'messages': [{'role': 'user', 'content': question}]})
    used = [d['content'] for d in state['documents']]
    if used != context:
        raise ValueError('Reference context exceeds the generator context budget')
    return {'answer': state['final_answer'], 'sources': state['documents'],
            'retrieval_context': used}


def run(question: str) -> dict:
    """Call /query with a fresh conversation and reject backend error responses.

    Application costs are unavailable until every billed stage is instrumented.
    A guardrail block is a completed response, while a guardrail error is not.
    """
    response = requests.post(os.getenv('EVAL_API_URL', 'http://localhost:8000/query'),
        json={'q': question, 'thread_id': f'deepeval_{uuid.uuid4().hex}'}, timeout=120)
    response.raise_for_status()
    data = response.json()
    if data.get('status') == 'error' or data.get('guardrail_decision') == 'error':
        raise ValueError('Backend reported an execution error')
    sources = data.get('sources')
    if not isinstance(sources, list):
        raise ValueError('Backend sources must be a list')
    contexts = [s.get('content') if isinstance(s, dict) else s for s in sources]
    return {'answer': data.get('answer'), 'retrieval_context': contexts,
            'sources': sources, 'guardrail_decision': data.get('guardrail_decision'),
            'plan': data.get('thought_process'), 'cost_usd': None,
            'input_tokens': None, 'output_tokens': None}

