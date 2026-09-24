"""Fuse dense/BM25 candidates, rerank, and assign citation IDs."""
import logfire
from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents


def retrieve_node(state: AgentState):
    """Preserve full candidate objects through both retrieval stages."""
    with logfire.span('Knowledge Retrieval'):
        candidates = search_enterprise_knowledge(state['current_query'], limit=15, prefetch_limit=40)
        with logfire.span('Semantic Reranking'):
            selected = rerank_documents(state['current_query'], candidates, top_n=5)
    documents = [{**doc, 'id':i} for i,doc in enumerate(selected, 1)]
    return {'documents':documents, 'status':f'Found {len(documents)} document chunks.',
            'plan':state['plan'] + ['Context Retrieved', 'Dense + BM25 → RRF → FlashRank']}
