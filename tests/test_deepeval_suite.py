"""Evaluate harness integrity; these tests do not claim application quality."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
from evals.deepeval_suite import evaluate as e, project_adapter as a, metrics as m


class DeepEvalSuiteTests(unittest.TestCase):
    def setUp(self):
        self.cfg = e.read_json(e.ROOT / 'config.json')
        self.cfg['request_delay_s'] = 0
        self.case = e.read_json(e.ROOT / 'goldens.json')[0]
        self.safety = e.read_json(e.ROOT / 'safety.json')[0]
        self.output = dict(answer='actual answer', retrieval_context=['actual evidence'], sources=[], guardrail_decision='allow')

    def test_api_adapter_isolates_threads_and_rejects_error(self):
        response = Mock()
        response.json.return_value = dict(answer='answer', sources=[{'content':'actual','point_id':'x'}], guardrail_decision='allow')
        with patch.object(a.requests, 'post', return_value=response) as post:
            result = a.run('question')
            a.run('question')
            self.assertEqual(result['retrieval_context'], ['actual'])
            self.assertNotEqual(post.call_args_list[0].kwargs['json']['thread_id'], post.call_args_list[1].kwargs['json']['thread_id'])
            self.assertIsNone(result['cost_usd'])
            response.json.return_value['status'] = 'error'
            with self.assertRaises(ValueError): a.run('question')

    def test_all_levels_reuse_api_but_isolate_generator(self):
        def score(level, case, output, *args):
            return [dict(level=level,case_id=case['id'],metric='fixture',score=1.,status='ok',passed=True,direction='higher')]
        with patch.object(a,'run',return_value=self.output) as api, patch.object(a,'retrieve',return_value=self.output) as retriever, patch.object(a,'generate',return_value=self.output) as generator, patch.object(e,'score_case',side_effect=score):
            report=e.run_suite(['retriever','generator','pipeline','quality','safety','ops'], [self.case], [self.safety], self.cfg, Mock())
        self.assertEqual(api.call_count, 2)
        retriever.assert_called_once_with(self.case['input'])
        generator.assert_called_once_with(self.case['input'], self.case['gold_context'])
        self.assertEqual(report['operations']['requests'], 1)
        self.assertEqual(len(report['requests']), 4)

    def test_generator_uses_production_node_and_detects_truncation(self):
        from app.agents.nodes import responder
        node = Mock(return_value={'final_answer':'answer','documents':[{'content':'verified'}]})
        with patch.object(responder,'generate_node',node):
            self.assertEqual(a.generate('question',['verified'])['retrieval_context'], ['verified'])
            self.assertEqual(node.call_args.args[0]['documents'][0]['content'],'verified')
            node.return_value['documents'] = []
            with self.assertRaises(ValueError): a.generate('question',['verified'])

    def test_metric_errors_empty_context_and_guardrail_decision(self):
        with patch.object(m,'build_metric',side_effect=RuntimeError()):
            row=m.score_case('generator', self.case, self.output, self.cfg, Mock(), ['faithfulness'])[0]
        self.assertEqual(row['status'],'error')
        self.assertIsNone(row['score'])
        empty={**self.output, 'retrieval_context':[]}
        judge=Mock()
        rows=m.score_case('pipeline',self.case,empty,self.cfg,judge,['faithfulness','contextual_relevancy'])
        self.assertEqual([r['status'] for r in rows],['ok','unscorable'])
        judge.assert_not_called()
        row=m.score_case('safety',self.safety,self.output,self.cfg,judge,['guardrail_decision'])[0]
        self.assertEqual(row['score'],0)

    def test_cost_and_reliability_are_honest(self):
        result=e.operations([dict(success=True,latency_s=1),dict(success=False,latency_s=3)],self.cfg)
        self.assertEqual(result['reliability'],.5)
        self.assertIsNone(result['mean_cost_usd'])
        self.assertFalse(result['passed'])
        self.cfg['ops']['max_mean_cost_usd']=.01
        self.assertFalse(e.operations([dict(success=True,latency_s=1)],self.cfg)['passed'])

    def test_validation_rejects_duplicate_and_blank_context(self):
        with self.assertRaises(ValueError): e.validate_cases([self.case,self.case])
        with self.assertRaises(ValueError): e.validate_cases([{**self.case,'gold_context':['']}])

    def test_regression_requires_matching_workload_and_complete_scores(self):
        meta={key:'same' for key in ('dataset_hash','config_hash','levels','selected_metrics','deepeval_version','suite_hash')}
        old={'metadata':meta,'summary':{'safety.toxicity':{'mean':1.,'scored':1,'total':1,'direction':'higher'}}}
        new=copy.deepcopy(old)
        new['summary']['safety.toxicity']['mean']=.8
        self.assertTrue(e.compare(old,new,self.cfg)[0]['regressed'])
        new['metadata']['dataset_hash']='different'
        with self.assertRaises(ValueError): e.compare(old,new,self.cfg)

    def test_trace_sampling_includes_failures_in_operations(self):
        traces=[dict(trace_id='a',input='q',answer='a',retrieval_context=['ctx'],route='rag',success=True,latency_s=1),
                dict(trace_id='b',route='rag',success=False,latency_s=2)]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'traces.jsonl'
            path.write_text('\n'.join(json.dumps(t) for t in traces))
            judge=Mock()
            report=e.run_traces(path,0,self.cfg,judge)
        judge.assert_not_called()
        self.assertEqual(report['records'],[])
        self.assertEqual(report['operations']['reliability'],.5)

    def test_real_metric_classes_accept_custom_judge(self):
        from evals.deepeval_suite.judge import GroqJudge
        with patch.object(GroqJudge,'load_model',return_value=Mock()):
            judge=GroqJudge('fixture',delay=0)
            for name in self.cfg['thresholds']:
                self.assertIsNotNone(m.build_metric(name,self.cfg,judge))

    def test_judge_validates_schema_and_tracks_usage(self):
        from evals.deepeval_suite.judge import GroqJudge
        from pydantic import BaseModel, ValidationError
        class Verdict(BaseModel):
            verdict: str
        response=SimpleNamespace(usage=SimpleNamespace(prompt_tokens=3,completion_tokens=2),
            choices=[SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content='{"verdict":"yes"}'))])
        client=Mock()
        client.chat.completions.create.return_value=response
        with patch.object(GroqJudge,'load_model',return_value=client):
            judge=GroqJudge('fixture',delay=0)
            self.assertEqual(judge.generate('prompt',Verdict).verdict,'yes')
            self.assertEqual(judge.usage['input_tokens'],3)
            response.choices[0].message.content='{"other":"missing field"}'
            with self.assertRaises(ValidationError): judge.generate('prompt',Verdict)
            judge.close()
            client.close.assert_called_once()

    def test_reference_passages_exist_in_sources(self):
        from bs4 import BeautifulSoup
        for case in e.read_json(e.ROOT/'goldens.json'):
            path=e.PROJECT/case['source']
            raw=path.read_text()
            text=BeautifulSoup(raw,'html.parser').get_text(' ') if path.suffix=='.html' else raw
            normalized=' '.join(text.split())
            for passage in case['gold_context']:
                self.assertIn(' '.join(passage.split()),normalized)
