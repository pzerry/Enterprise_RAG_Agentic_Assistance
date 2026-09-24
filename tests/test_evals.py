"""Regression tests for collection integrity, error accounting and metric inputs."""
import asyncio
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, AsyncMock, patch
from evals import pipeline, guardrails_eval as guards, metrics


def dataset():
    return {'rag_samples':[{'id':1,'question':'Why optimize?', 'reference':'Improve speed.',
        'relevant_contexts':['GOLD ONLY'], 'expected_tools':['retrieve_documents']}], 'guardrails_samples':[]}


class EvalTests(unittest.TestCase):
    def test_collection_preserves_complete_answer_and_evidence(self):
        d=dataset()
        response=Mock(json=lambda:{'answer':'a'*1200,'sources':['b'*1500]*5,
                                   'thought_process':['Intent: Technical']})
        with patch.object(pipeline.requests,'post',return_value=response) as post:
            result=pipeline.run_pipeline(d)
            pipeline.run_pipeline(d)
        row=result['rag_samples'][0]
        self.assertEqual(len(row['actual_response']),1200)
        self.assertEqual(len(row['actual_contexts']),5)
        self.assertNotEqual(post.call_args_list[0].kwargs['json']['thread_id'],post.call_args_list[1].kwargs['json']['thread_id'])
        self.assertNotIn('actual_response',d['rag_samples'][0])

    def test_errors_never_invent_evidence(self):
        for response in [Mock(json=lambda:{'status':'error','answer':'Sorry'}),
                         Mock(json=lambda:{'answer':None})]:
            with patch.object(pipeline.requests,'post',return_value=response):
                row=pipeline.run_pipeline(dataset())['rag_samples'][0]
            self.assertEqual(row['eval_status'],'error')
            self.assertEqual(row['actual_contexts'],[])
        with patch.object(pipeline.requests,'post',side_effect=ConnectionError()):
            row=pipeline.run_pipeline(dataset())['rag_samples'][0]
        self.assertEqual(row['actual_contexts'],[])

    def test_guard_failures_are_unscored(self):
        with patch.object(guards.requests,'post',side_effect=ConnectionError()):
            result=guards.run_guardrails_eval([{'id':1,'input':'hello','expected_blocked':False}])
        summary=guards.compute_guardrails_metrics(result)
        self.assertEqual(summary['errors'],1)
        self.assertEqual(summary['evaluated'],0)
        self.assertIsNone(summary['accuracy'])
        self.assertFalse(guards._is_blocked({'guardrail_decision':'handled'}))
        self.assertTrue(guards._is_blocked({'guardrail_decision':'block'}))
        with self.assertRaises(ValueError): guards._is_blocked({'thought_process':['Intent: Guardrails Fired']})

    def test_guard_confusion_matrix(self):
        summary=guards.compute_guardrails_metrics([{'result':r} for r in ['TP','TN','FP','FN','ERROR']])
        self.assertEqual(summary['accuracy'],.5)
        self.assertEqual(summary['precision'],.5)
        self.assertEqual(summary['recall'],.5)
        self.assertEqual(summary['errors'],1)

    def test_scoring_passes_full_evidence_and_closes_client(self):
        d=dataset();row=d['rag_samples'][0]
        row.update(eval_status='ok',actual_response='answer'*200,actual_contexts=['evidence'*300]*5)
        fake=Mock(ascore=AsyncMock(return_value=SimpleNamespace(value=1.0)))
        client=SimpleNamespace(close=AsyncMock())
        with patch.object(metrics,'_build_judge',return_value=(object(),client)),patch('ragas.metrics.collections.Faithfulness',return_value=fake):
            result=asyncio.run(metrics.run_all_metrics(d,selected=['faithfulness']))
        self.assertEqual(fake.ascore.call_args.kwargs['retrieved_contexts'],row['actual_contexts'])
        self.assertEqual(fake.ascore.call_args.kwargs['response'],row['actual_response'])
        client.close.assert_awaited_once()
        self.assertEqual(result['faithfulness'].iloc[0]['status'],'ok')

    def test_missing_contexts_not_replaced_and_failures_retained(self):
        d=dataset();d['rag_samples'][0].update(eval_status='ok',actual_response='answer',actual_contexts=[],actual_tools_called=['retrieve_documents'])
        d['rag_samples'].append({**d['rag_samples'][0], 'id':2,'eval_status':'error'})
        with patch.object(metrics,'_build_judge') as build:
            result=asyncio.run(metrics.run_all_metrics(d,selected=['faithfulness','context_recall','tool_correctness']))
        build.assert_not_called()
        self.assertEqual(list(result['faithfulness']['status']),['unscorable','error'])
        self.assertEqual(result['context_recall'].iloc[0]['context_recall'],0)
        self.assertEqual(result['tool_correctness'].iloc[0]['tool_correctness'],1)

    def test_judge_exception_remains_error(self):
        d=dataset();d['rag_samples'][0].update(eval_status='ok',actual_response='answer',actual_contexts=['evidence'])
        with patch.object(metrics,'_build_judge',side_effect=RuntimeError('failed')):
            r=asyncio.run(metrics.run_all_metrics(d,selected=['faithfulness']))
        self.assertEqual(r['faithfulness'].iloc[0]['status'],'error')


if __name__=='__main__': unittest.main()
