"""Local BM25 document/query vectors; Qdrant applies corpus IDF at search time."""
from functools import lru_cache
from fastembed import SparseTextEmbedding
from qdrant_client import models

SPARSE_MODEL = 'Qdrant/bm25'


@lru_cache(maxsize=1)
def _get_sparse_model():
    """Load the BM25 tokenizer lazily; no Gemini or Groq calls are made."""
    return SparseTextEmbedding(model_name=SPARSE_MODEL)


def embed_sparse_documents(texts: list[str]) -> list[models.SparseVector]:
    """Encode document term frequencies/lengths using BM25's document path."""
    if not texts:
        return []
    return [models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
            for e in _get_sparse_model().embed(texts)]


def embed_sparse_query(query: str) -> models.SparseVector:
    """Encode query terms with query_embed, not the document embedding method."""
    e = next(_get_sparse_model().query_embed(query))
    return models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
