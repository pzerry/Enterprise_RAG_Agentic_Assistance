"""FlashRank reranking that retains source identity and retrieval scores."""
import math
import logfire
from flashrank import Ranker, RerankRequest

_ranker = None


def _get_ranker():
    """Load the existing TinyBERT cross-encoder once on first use."""
    global _ranker
    if _ranker is None:
        _ranker = Ranker(model_name='ms-marco-TinyBERT-L-2-v2', cache_dir='/tmp/flashrank')
    return _ranker


def rerank_documents(query: str, documents: list[dict], top_n: int = 5) -> list[dict]:
    """Return top candidates with FlashRank scores, leaving inputs unchanged.

    On reranker failure retain the hybrid order, metadata, and an explicit
    fallback marker. Do not silently present an RRF score as a reranker score.
    """
    if top_n < 1:
        raise ValueError('top_n must be positive')
    if not documents:
        return []
    try:
        request = RerankRequest(query=query, passages=[{'id':i, 'text':d['content']}
                                                     for i,d in enumerate(documents)])
        ranked = _get_ranker().rerank(request)
        seen, results = set(), []
        for item in ranked[:top_n]:
            i, score = int(item['id']), float(item['score'])
            if i < 0 or i >= len(documents) or i in seen or not math.isfinite(score):
                raise ValueError('Invalid reranker result')
            seen.add(i)
            results.append({**documents[i], 'rerank_score':score, 'rerank_status':'ok'})
        if len(results) != min(top_n, len(documents)):
            raise ValueError('Incomplete reranker results')
        return results
    except Exception as exc:
        logfire.warning('Reranking failed; retaining hybrid order: {error_type}', error_type=type(exc).__name__)
        return [{**d, 'rerank_score':None, 'rerank_status':'fallback'} for d in documents[:top_n]]
