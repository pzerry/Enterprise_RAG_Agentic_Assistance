"""Qdrant dense + sparse candidate retrieval with reciprocal rank fusion."""
import logfire
from qdrant_client import QdrantClient, models
from app.config import settings
from app.services.retrieval.embeddings import embed_query
from app.services.retrieval.sparse_embeddings import embed_sparse_query

client = None


def _get_client():
    """Connect lazily so importing retrieval does not contact the database."""
    global client
    if client is None:
        client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    return client


def search_enterprise_knowledge(query: str, limit: int = 15, prefetch_limit: int = 40) -> list[dict]:
    """Fuse up to 40 dense and BM25 matches into 15 candidates by default.

    Returned hybrid_score is an RRF score, not cosine similarity or confidence.
    Preserve point identity and metadata. Connection/schema errors propagate to
    the API instead of being disguised as a successful search with no evidence.
    """
    if not query.strip():
        raise ValueError('Search query must not be empty')
    if limit < 1 or prefetch_limit < limit:
        raise ValueError('Require prefetch_limit >= limit >= 1')
    db = _get_client()
    # Validate before paying for a query embedding.
    params = db.get_collection(settings.QDRANT_COLLECTION).config.params
    if not isinstance(params.vectors, dict) or 'dense' not in params.vectors or 'sparse' not in (params.sparse_vectors or {}):
        raise ValueError('Collection is not hybrid. Ingest into enterprise_rag_hybrid_v1 first; do not wipe the old index.')
    if (params.vectors['dense'].size != settings.EMBEDDING_DIM
            or params.vectors['dense'].distance != models.Distance.COSINE
            or params.sparse_vectors['sparse'].modifier != models.Modifier.IDF):
        raise ValueError('Collection dense dimensions or sparse IDF configuration do not match')
    dense = embed_query(query)
    sparse = embed_sparse_query(query)
    prefetch = [models.Prefetch(query=dense, using='dense', limit=prefetch_limit)]
    # Stopword-only inputs may have no lexical terms; dense search still works.
    if sparse.indices:
        prefetch.append(models.Prefetch(query=sparse, using='sparse', limit=prefetch_limit))
    response = db.query_points(collection_name=settings.QDRANT_COLLECTION,
        prefetch=prefetch, query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit, with_payload=True)
    results = []
    for point in response.points:
        payload = point.payload or {}
        if not payload.get('text'):
            continue
        results.append(dict(point_id=str(point.id), content=payload['text'],
            source=payload.get('source', 'Unknown'), source_type=payload.get('source_type'),
            document_id=payload.get('document_id'), version=payload.get('version'),
            model=payload.get('model'), hybrid_score=float(point.score)))
    logfire.info('Hybrid retrieval returned {count} candidates', count=len(results))
    return results
