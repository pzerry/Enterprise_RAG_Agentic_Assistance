import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock

import logfire
from qdrant_client import QdrantClient, models
from app.config import settings
from app.ingestion.chunking.splitter import chunk_text
from app.ingestion import processor as p
from app.services.retrieval import embeddings as e

logfire.configure(send_to_logfire=False, console=False)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.db = QdrantClient(':memory:')
        self.addCleanup(self.db.close)
        self.db.create_collection(settings.QDRANT_COLLECTION, vectors_config={'dense': models.VectorParams(size=3, distance=models.Distance.COSINE)}, sparse_vectors_config={'sparse': models.SparseVectorParams(modifier=models.Modifier.IDF)})
        for target, value in [('EMBEDDING_DIM', 3), ('EMBEDDING_STATE', str(self.root/'budget.sqlite'))]:
            patcher = patch.object(settings, target, value)
            patcher.start(); self.addCleanup(patcher.stop)
        for target, value in [('qdrant_client', self.db), ('PROCESSED_DATA_DIR', str(self.root/'processed'))]:
            patcher = patch.object(p, target, value)
            patcher.start(); self.addCleanup(patcher.stop)
        sparse_patch = patch.object(p, 'embed_sparse_documents', side_effect=lambda chunks:
            [models.SparseVector(indices=[1], values=[1.0]) for _ in chunks])
        sparse_patch.start(); self.addCleanup(sparse_patch.stop)
        self.file = self.root/'example.txt'
        self.file.write_text('long document ' * 500)

    def count(self):
        return self.db.count(settings.QDRANT_COLLECTION, exact=True).count

    def test_payload_indexes_created_before_ingestion(self):
        """Simulate cloud strict mode: ingestion requires both indexed fields."""
        db = Mock()
        schema = {}
        db.collection_exists.return_value = True
        db.get_collection.side_effect = lambda name: SimpleNamespace(
            payload_schema=schema,
            config=SimpleNamespace(params=SimpleNamespace(vectors={'dense': models.VectorParams(
                size=3, distance=models.Distance.COSINE)}, sparse_vectors={'sparse': models.SparseVectorParams(modifier=models.Modifier.IDF)})))
        def create(**kwargs):
            self.assertTrue(kwargs['wait'])
            self.assertEqual(kwargs['field_schema'], models.PayloadSchemaType.KEYWORD)
            schema[kwargs['field_name']] = SimpleNamespace(data_type=models.PayloadSchemaType.KEYWORD)
        db.create_payload_index.side_effect = create
        def scan(*args):
            self.assertEqual(set(schema), {'document_id', 'version'})
            return dict(indexed=1, skipped=0, failed=0)
        with patch.object(p, 'qdrant_client', db), patch.object(p, 'process_directory', side_effect=scan):
            p.run_universal_ingestion(str(self.root), 'test')
            p.run_universal_ingestion(str(self.root), 'test')
        self.assertEqual(db.create_payload_index.call_count, 2)

    def test_index_setup_failure_prevents_embedding(self):
        db = Mock()
        db.collection_exists.return_value = True
        db.get_collection.return_value = SimpleNamespace(payload_schema={},
            config=SimpleNamespace(params=SimpleNamespace(vectors={'dense': models.VectorParams(
                size=3, distance=models.Distance.COSINE)}, sparse_vectors={'sparse': models.SparseVectorParams(modifier=models.Modifier.IDF)})))
        db.create_payload_index.side_effect = RuntimeError('Index creation failed')
        with patch.object(p, 'qdrant_client', db), patch.object(p, 'embed_texts') as embed:
            with self.assertRaisesRegex(RuntimeError, 'Index creation failed'):
                p.run_universal_ingestion(str(self.root), 'test')
            embed.assert_not_called()

    def test_qdrant_error_body_is_not_truncated(self):
        body = ('missing index ' + 'x' * 500 + ' document_id').encode()
        exc = p.UnexpectedResponse(400, 'Bad Request', body, {})
        self.assertIn(body.decode(), p._error_text(exc))

    def test_chunks_preserve_content_and_bound(self):
        for text in ['a'*5000, 'word '*1200, '你😀好'*1700, 'a\r\n\r\nb']:
            chunks = chunk_text(text)
            self.assertTrue(all(0 < len(c) <= 1500 for c in chunks))
            self.assertEqual(''.join(text.split()), ''.join(''.join(chunks).split()))
        self.assertEqual(chunk_text('  '), [])
        with self.assertRaises(ValueError): chunk_text('a', 0)

    def test_rerun_skip_and_changed_document_cleanup(self):
        with patch.object(p, 'embed_texts', side_effect=lambda chunks: [[1.,0.,0.]]*len(chunks)) as embed:
            self.assertEqual(p.process_file(str(self.file), self.file.name, 'test'), 'indexed')
            first_count = self.count()
            self.assertGreater(first_count, 1)
            self.assertEqual(p.process_file(str(self.file), self.file.name, 'test'), 'skipped')
            self.assertEqual(embed.call_count, 1)
            self.file.write_text('changed short document')
            self.assertEqual(p.process_file(str(self.file), self.file.name, 'test'), 'indexed')
            self.assertEqual(self.count(), 1)

    def test_failure_retry_and_missing_remote_points(self):
        with patch.object(p, 'embed_texts', side_effect=RuntimeError('failed')):
            self.assertEqual(p.process_file(str(self.file), self.file.name, 'test'), 'failed')
        manifest = next((self.root/'processed/test').glob('*.json'))
        self.assertEqual(json.loads(manifest.read_text())['status'], 'failed')
        self.assertEqual(self.count(), 0)
        with patch.object(p, 'embed_texts', side_effect=lambda chunks: [[1.,0.,0.]]*len(chunks)):
            self.assertEqual(p.process_file(str(self.file), self.file.name, 'test'), 'indexed')
            ids = json.loads(manifest.read_text())['point_ids']
            self.db.delete(settings.QDRANT_COLLECTION, models.PointIdsList(points=[ids[0]]), wait=True)
            self.assertEqual(p.process_file(str(self.file), self.file.name, 'test'), 'indexed')
            self.assertEqual(self.count(), len(ids))

    def test_partial_upsert_retry_is_idempotent(self):
        original = self.db.upsert
        def write_then_fail(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError('Acknowledgement lost')
        with patch.object(p, 'embed_texts', side_effect=lambda chunks: [[1.,0.,0.]]*len(chunks)):
            with patch.object(self.db, 'upsert', side_effect=write_then_fail):
                self.assertEqual(p.process_file(str(self.file), self.file.name, 'test'), 'failed')
            before = self.count()
            self.assertEqual(p.process_file(str(self.file), self.file.name, 'test'), 'indexed')
            self.assertEqual(self.count(), before)

    def test_mismatched_vectors_refused(self):
        with patch.object(p, 'embed_texts', return_value=[]):
            self.assertEqual(p.process_file(str(self.file), self.file.name, 'test'), 'failed')
            self.assertEqual(self.count(), 0)

    def test_budget_persists_and_waits(self):
        clock = [100000.]
        def sleep(delay): clock[0] += delay
        with patch.object(settings, 'EMBEDDING_TPM', 100), patch.object(e.time, 'time', side_effect=lambda:clock[0]), patch.object(e.time, 'sleep', side_effect=sleep) as waits:
            e._reserve(60)
            e._reserve(60)
            self.assertGreaterEqual(clock[0], 100061.)
            self.assertEqual(waits.call_count, 1)
            with self.assertRaises(ValueError): e._reserve(101)
        with patch.object(settings, 'EMBEDDING_RPD', 1), patch.object(e.time, 'time', return_value=clock[0]):
            with self.assertRaises(e.QuotaExhausted): e._reserve(1)

    def test_rpm_budget(self):
        clock = [100000.]
        with patch.object(settings, 'EMBEDDING_RPM', 1), patch.object(e.time, 'time', side_effect=lambda:clock[0]), patch.object(e.time, 'sleep', side_effect=lambda delay:clock.__setitem__(0,clock[0]+delay)):
            e._reserve(1); e._reserve(1)
            self.assertGreaterEqual(clock[0], 100061.)

    def test_retry_delay_and_cache(self):
        class RateLimit(Exception):
            code = 429
            response_json = {'error': {'details': [{'retryDelay': '72s'}]}}
        mock_client = SimpleNamespace(models=SimpleNamespace())
        result = SimpleNamespace(embeddings=[SimpleNamespace(values=[1.,0.,0.])])
        with patch.object(e, '_client', mock_client), patch.object(mock_client.models, 'embed_content', create=True, side_effect=[RateLimit(), result]) as api, patch.object(e, '_reserve') as reserve, patch.object(e.time, 'sleep') as sleep:
            self.assertEqual(e.embed_texts(['hello']), [[1.,0.,0.]])
            self.assertGreaterEqual(sleep.call_args.args[0], 72)
            self.assertEqual(reserve.call_count, 2)
            self.assertEqual(e.embed_texts(['hello']), [[1.,0.,0.]])
            self.assertEqual(api.call_count, 2)

    def test_real_sdk_serializes_gemini_request(self):
        """Exercise real SDK validation; mock only transport, not embed_content.

        Unsupported Vertex-only options must fail here before any live call.
        Check both document and query paths with the Gemini API serializer.
        """
        sdk = e.genai.Client(api_key='test-key', vertexai=False)
        self.addCleanup(sdk.close)
        response = SimpleNamespace(
            body=json.dumps({'embeddings': [{'values': [1., 0., 0.]}]}),
            headers={},
        )
        with patch.object(e, '_client', sdk), patch.object(e, '_reserve'), patch.object(
            sdk._api_client, 'request', return_value=response
        ) as transport:
            self.assertEqual(e.embed_texts(['document']), [[1., 0., 0.]])
            self.assertEqual(e.embed_query('query'), [1., 0., 0.])
            self.assertEqual(transport.call_count, 2)
            for call in transport.call_args_list:
                body = call.args[2]
                self.assertNotIn('autoTruncate', json.dumps(body))
                self.assertEqual(body['requests'][0]['outputDimensionality'], 3)
            with self.assertRaisesRegex(ValueError, 'too large'):
                e.embed_texts(['x' * 8000])
            self.assertEqual(transport.call_count, 2)

    def test_daily_quota_stops_without_retry(self):
        class DailyLimit(Exception):
            code=429
            response_json={'error':{'details':[{'quotaId':'RequestsPerDay'}]}}
        mock_client=SimpleNamespace(models=SimpleNamespace())
        with patch.object(e, '_client', mock_client), patch.object(mock_client.models, 'embed_content', create=True, side_effect=DailyLimit()), patch.object(e, '_reserve'), patch.object(e.time, 'sleep') as sleep:
            with self.assertRaises(e.QuotaExhausted): e.embed_query('hello')
            sleep.assert_not_called()


if __name__ == '__main__':
    unittest.main()
