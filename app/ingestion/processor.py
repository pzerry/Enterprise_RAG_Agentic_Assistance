"""Resumable ingestion. Run with python -m app.ingestion.processor PATH SOURCE."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import uuid

import logfire
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from app.config import settings
from app.services.retrieval.sparse_embeddings import embed_sparse_documents, SPARSE_MODEL
from app.services.retrieval.embeddings import embed_texts, get_embedding_dim, QuotaExhausted
from app.ingestion.chunking.splitter import chunk_text, CHUNKER_VERSION
from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.text import parse_text

PROCESSED_DATA_DIR = 'processed_data'
qdrant_client = None


def client():
    """Lazily construct and reuse the configured Qdrant client.

    Importing this module does not connect to Qdrant. Tests can supply an
    in-memory client through the module's ``qdrant_client`` variable.
    """
    global qdrant_client
    if qdrant_client is None:
        qdrant_client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    return qdrant_client


def save_processed_locally(data, source_type, filename):
    """Atomically save a document manifest and return its filesystem path.

    Write a temporary JSON file before replacing the destination, so readers
    never see partially written JSON. The processor passes the stable document
    ID as ``filename`` to avoid collisions between equally named source files.
    ``status`` distinguishes parsed data from successfully indexed data.
    """
    folder = Path(PROCESSED_DATA_DIR) / source_type
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f'{filename}.json'
    temp = dest.with_suffix(f'.{uuid.uuid4().hex}.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, dest)
    return str(dest)


def _filter(document_id, version=None):
    """Build a Qdrant payload filter for one document, optionally one version.

    The document identity scopes cleanup so changing one source cannot delete
    chunks belonging to a different file.
    """
    conditions = [models.FieldCondition(key='document_id', match=models.MatchValue(value=document_id))]
    if version:
        conditions.append(models.FieldCondition(key='version', match=models.MatchValue(value=version)))
    return models.Filter(must=conditions)


def ensure_payload_indexes(db):
    """Create missing keyword indexes before filtering on document versions.

    Qdrant Cloud strict mode can reject cleanup on unindexed fields. Check both
    new and existing collections, and wait for index creation to finish before
    processing documents. Existing keyword indexes are left unchanged; an
    incompatible index raises an error before spending embedding quota.
    """
    schema = db.get_collection(settings.QDRANT_COLLECTION).payload_schema or {}
    for field in ('document_id', 'version'):
        existing = schema.get(field)
        if existing is not None:
            if existing.data_type != models.PayloadSchemaType.KEYWORD:
                raise ValueError(f'Payload index {field!r} must have type keyword')
            continue
        db.create_payload_index(
            collection_name=settings.QDRANT_COLLECTION,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )
        logfire.info('Created Qdrant payload index: {field}', field=field)


def _error_text(exc):
    """Include Qdrant's complete response body instead of its truncated str()."""
    if isinstance(exc, UnexpectedResponse):
        return f'Qdrant HTTP {exc.status_code}: {exc.content.decode("utf-8", errors="replace")}'
    return str(exc)


def process_file(file_path, filename, source_type):
    """Parse, chunk, embed, and index one file with resumable state.

    A stable document ID uses the absolute source path and source type. A version
    hash includes file bytes, chunking settings, model, dimension, and retrieval
    formatting. Skip a successful unchanged version only after checking that
    its recorded point IDs still exist in the configured Qdrant collection.

    Save a pending manifest before embedding. Write deterministic point IDs with
    acknowledged upserts, then remove older versions and mark the manifest
    indexed. A retry overwrites the same point IDs rather than adding duplicates.
    During partial updates, old/new chunks may coexist until a retry completes.

    Args:
        file_path: Path to the source document.
        filename: Display filename, including its extension.
        source_type: Source grouping, such as 'true' or 'noisy'.

    Returns:
        'indexed', 'skipped', or 'failed' for the run summary.

    Raises:
        QuotaExhausted: Stop the entire run when further requests cannot proceed.

    Moving/copying a source to a new path creates a distinct document identity.
    Do not run concurrent ingestions that update the same source document.
    """
    data = None
    document_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(Path(file_path).resolve()) + '|' + source_type))
    try:
        raw_hash = hashlib.sha256(Path(file_path).read_bytes()).hexdigest()
        version = hashlib.sha256(json.dumps([raw_hash, CHUNKER_VERSION, settings.CHUNK_SIZE,
            settings.EMBEDDING_MODEL, get_embedding_dim(), 'retrieval-prefix-v1', SPARSE_MODEL, 'hybrid-v1']).encode()).hexdigest()
        manifest = Path(PROCESSED_DATA_DIR) / source_type / f'{document_id}.json'
        if manifest.exists():
            previous = json.loads(manifest.read_text())
            if previous.get('status') == 'indexed' and previous.get('version') == version and previous.get('collection') == settings.QDRANT_COLLECTION:
                ids = previous['point_ids']
                existing = []
                for i in range(0, len(ids), 128):
                    existing.extend(client().retrieve(settings.QDRANT_COLLECTION, ids[i:i+128], with_payload=False, with_vectors=False))
                if len(existing) == len(ids):
                    logfire.info('Skipping unchanged indexed document: {file}', file=filename)
                    return 'skipped'
        ext = Path(filename).suffix.lower()
        loader = {'.pdf': parse_pdf, '.txt': parse_text, '.html': parse_html, '.htm': parse_html}.get(ext)
        if ext in ('.docx', '.pptx'):
            from app.ingestion.loaders.office import parse_office
            loader = parse_office
        if loader is None:
            return 'skipped'
        chunks = chunk_text(loader(file_path))
        if not chunks:
            raise ValueError('No extractable text; document was not indexed')
        ids = [str(uuid.uuid5(uuid.UUID(document_id), f'{version}:{i}')) for i in range(len(chunks))]
        data = dict(filename=filename, source_path=str(Path(file_path).resolve()), source_type=source_type,
                    document_id=document_id, version=version, collection=settings.QDRANT_COLLECTION,
                    model=settings.EMBEDDING_MODEL, chunks=chunks, point_ids=ids, status='pending')
        save_processed_locally(data, source_type, document_id)
        vectors = embed_texts(chunks)
        if len(vectors) != len(chunks) or any(len(v) != get_embedding_dim() for v in vectors):
            raise ValueError('Embedding count/dimension mismatch; refusing incomplete indexing')
        sparse_vectors = embed_sparse_documents(chunks)
        if len(sparse_vectors) != len(chunks):
            raise ValueError('Sparse embedding count mismatch')
        points = [models.PointStruct(id=pid, vector={'dense':v, 'sparse':sparse}, payload=dict(text=c, source=filename,
            source_type=source_type, document_id=document_id, version=version, model=settings.EMBEDDING_MODEL, sparse_model=SPARSE_MODEL))
            for pid, c, v, sparse in zip(ids, chunks, vectors, sparse_vectors)]
        for i in range(0, len(points), 64):
            client().upsert(settings.QDRANT_COLLECTION, points=points[i:i+64], wait=True)
        # Remove old versions only after all new vectors are acknowledged.
        stale = _filter(document_id)
        stale.must_not = [models.FieldCondition(key='version', match=models.MatchValue(value=version))]
        client().delete(settings.QDRANT_COLLECTION, points_selector=models.FilterSelector(filter=stale), wait=True)
        data['status'] = 'indexed'
        save_processed_locally(data, source_type, document_id)
        logfire.info('Indexed {count} chunks: {file}', count=len(points), file=filename)
        return 'indexed'
    except Exception as exc:
        if data:
            data['status'] = 'failed'
            data['error_type'] = type(exc).__name__
            save_processed_locally(data, source_type, document_id)
        logfire.error('Failed document {file}: {error}', file=filename, error=_error_text(exc))
        if isinstance(exc, QuotaExhausted):
            raise
        return 'failed'


def process_directory(dir_path, source_type):
    """Recursively ingest files in deterministic order and return status counts.

    Ordinary file failures are counted while remaining files continue. A quota
    exhaustion exception propagates immediately to stop further API requests.
    """
    counts = dict(indexed=0, skipped=0, failed=0)
    for path in sorted(Path(dir_path).rglob('*')):
        if path.is_file():
            counts[process_file(str(path), path.name, source_type)] += 1
    return counts


def run_universal_ingestion(base_dir, explicit_source_type=None, wipe=False):
    """Prepare the collection, route source folders, and return ingestion counts.

    Args:
        base_dir: Existing input directory.
        explicit_source_type: Optional label applied to the entire directory tree.
            Otherwise infer labels from top-level folder names.
        wipe: Explicitly delete the configured collection before rebuilding it.
            Cached embeddings remain reusable; manifests verify remote IDs.

    Returns:
        Counts of indexed, skipped, and failed documents.

    Raises:
        ValueError: Input directory or collection vector configuration is invalid.
        QuotaExhausted: Local/provider quota prevents continuing this run.

    Use a separate collection when changing embedding models, even when their
    dimensions match: equal dimensions do not mean compatible vector spaces.
    """
    if not Path(base_dir).is_dir():
        raise ValueError(f'Not a directory: {base_dir}')
    db = client()
    if wipe and db.collection_exists(settings.QDRANT_COLLECTION):
        db.delete_collection(settings.QDRANT_COLLECTION)
    if not db.collection_exists(settings.QDRANT_COLLECTION):
        db.create_collection(settings.QDRANT_COLLECTION,
            vectors_config={'dense':models.VectorParams(size=get_embedding_dim(), distance=models.Distance.COSINE)},
            sparse_vectors_config={'sparse':models.SparseVectorParams(modifier=models.Modifier.IDF)})
    params = db.get_collection(settings.QDRANT_COLLECTION).config.params
    vectors = params.vectors
    if (not isinstance(vectors, dict) or 'dense' not in vectors
            or vectors['dense'].size != get_embedding_dim()
            or vectors['dense'].distance != models.Distance.COSINE
            or 'sparse' not in (params.sparse_vectors or {})
            or params.sparse_vectors['sparse'].modifier != models.Modifier.IDF):
        raise ValueError('Hybrid schema required: named dense + sparse with IDF. Use a new collection; do not wipe the old index.')
    ensure_payload_indexes(db)
    counts = dict(indexed=0, skipped=0, failed=0)
    base = Path(base_dir)
    subdirs = sorted(p for p in base.iterdir() if p.is_dir())
    targets = [base] if explicit_source_type or not subdirs else subdirs
    for folder in targets:
        name = folder.name.lower()
        source = explicit_source_type or ('true' if 'true' in name else 'noisy' if 'noisy' in name else name)
        result = process_directory(str(folder), source)
        for key in counts:
            counts[key] += result[key]
    logfire.info('Ingestion summary: {counts}', counts=counts)
    return counts


def main():
    """Run the command-line entry point and return an honest process exit code.

    Print a status summary after a completed scan. Return 0 only when no files
    failed, or 1 when files failed, quota stopped the run, or setup failed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', nargs='?', default='DATA')
    parser.add_argument('source', nargs='?')
    parser.add_argument('--wipe', action='store_true', help='Delete the configured collection first')
    args = parser.parse_args()
    logfire.configure(service_name='enterprise-ingestion-service')
    try:
        counts = run_universal_ingestion(args.directory, args.source, args.wipe)
        print(json.dumps(counts))
        return 1 if counts['failed'] else 0
    except Exception as exc:
        logfire.error('Ingestion stopped: {error}', error=_error_text(exc))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
