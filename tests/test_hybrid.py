"""Exercise hybrid fusion, metadata, and evidence boundaries without paid APIs."""
import unittest
from unittest.mock import patch, Mock
from types import SimpleNamespace
from qdrant_client import QdrantClient, models
from app.config import settings
from app.services.retrieval import qdrant_service as q, ranking_service as r
from app.agents.nodes import responder
from evals import pipeline


class HybridTests(unittest.TestCase):
    def test_real_qdrant_fusion_preserves_both_search_branches(self):
        db = QdrantClient(':memory:')
        self.addCleanup(db.close)
        db.create_collection('hybrid', vectors_config={'dense': models.VectorParams(size=3, distance=models.Distance.COSINE)},
            sparse_vectors_config={'sparse': models.SparseVectorParams(modifier=models.Modifier.IDF)})
        db.upsert('hybrid', points=[models.PointStruct(id=i,
            vector={'dense': v, 'sparse': models.SparseVector(indices=[term], values=[1.])},
            payload={'text': f'passage {i}', 'source': f'{i}.pdf', 'document_id':f'doc{i}', 'version':'v1'})
            for i,v,term in [(1,[1.,0.,0.],10),(2,[0.,1.,0.],20),(3,[-1.,0.,0.],30)]])
        with patch.object(q,'client',db), patch.object(settings,'QDRANT_COLLECTION','hybrid'), patch.object(settings,'EMBEDDING_DIM',3), patch.object(q,'embed_query',return_value=[1.,0.,0.]), patch.object(q,'embed_sparse_query',return_value=models.SparseVector(indices=[30],values=[1.])):
            docs = q.search_enterprise_knowledge('exact term',limit=2,prefetch_limit=2)
        self.assertEqual({d['point_id'] for d in docs},{'1','3'})
        self.assertTrue(all(d['source'] and d['version']=='v1' and d['hybrid_score']>0 for d in docs))

    def test_old_schema_rejected_before_embedding(self):
        db=Mock()
        db.get_collection.return_value=SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=models.VectorParams(size=3,distance=models.Distance.COSINE))))
        with patch.object(q,'client',db), patch.object(q,'embed_query') as embed:
            with self.assertRaisesRegex(ValueError,'not hybrid'):
                q.search_enterprise_knowledge('question')
            embed.assert_not_called()

    def test_reranking_metadata_and_explicit_fallback(self):
        docs=[{'content':str(i),'point_id':str(i),'source':'doc','hybrid_score':.3} for i in range(3)]
        ranker=Mock()
        ranker.rerank.return_value=[{'id':2,'score':.9},{'id':0,'score':.4}]
        with patch.object(r,'_get_ranker',return_value=ranker):
            result=r.rerank_documents('query',docs,2)
            self.assertEqual([d['point_id'] for d in result],['2','0'])
            self.assertEqual(result[0]['hybrid_score'],.3)
            self.assertNotIn('rerank_score',docs[2])
            ranker.rerank.side_effect=RuntimeError('failure')
            result=r.rerank_documents('query',docs,2)
            self.assertEqual([d['point_id'] for d in result],['0','1'])
            self.assertEqual(result[0]['rerank_status'],'fallback')
            self.assertIsNone(result[0]['rerank_score'])

    def test_generator_sources_match_prompt_and_clear_conversation(self):
        docs=[{'id':1,'source':'test.pdf','content':'Supported evidence'},
              {'id':2,'source':'large.pdf','content':'x'*25000}]
        state={'current_query':'query','messages':[{'role':'user','content':'question'}],'documents':docs,'plan':[]}
        with patch.object(responder,'llm') as llm:
            llm.invoke.return_value=SimpleNamespace(content='Answer [1]')
            result=responder.generate_node(state)
            self.assertEqual(result['documents'],docs[:1])
            prompt=llm.invoke.call_args.args[0]
            self.assertIn('[1] SOURCE: test.pdf',prompt[1].content)
            self.assertNotIn('large.pdf',prompt[1].content)
            state['current_query']='CONVERSATIONAL'
            self.assertEqual(responder.generate_node(state)['documents'],[])
            llm.reset_mock()
            state.update(current_query='query',documents=[])
            self.assertEqual(responder.generate_node(state)['documents'],[])
            llm.invoke.assert_not_called()

    def test_evaluations_keep_metadata_and_score_text(self):
        sources=[{'id':1,'content':'full evidence','source':'doc.pdf','hybrid_score':.5}]
        response=Mock()
        response.json.return_value={'answer':'Answer [1]','sources':sources,'status':'Response generated.','thought_process':['Context Retrieved']}
        with patch.object(pipeline.requests,'post',return_value=response):
            sample=pipeline.run_pipeline({'rag_samples':[{'question':'test'}]})['rag_samples'][0]
        self.assertEqual(sample['eval_status'],'ok')
        self.assertEqual(sample['actual_contexts'],['full evidence'])
        self.assertEqual(sample['actual_sources'],sources)
