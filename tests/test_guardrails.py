"""Tests for the current async NeMo input-rail interface."""
import unittest
from unittest.mock import patch, AsyncMock, Mock
from nemoguardrails.rails.llm.options import RailStatus
from app.Guardrails import rails


class GuardrailTests(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        rails._rails = None

    async def test_no_initialization_fails_closed(self):
        rails._rails = None
        result = await rails.guard_input('Explain Kubernetes')
        self.assertEqual(result.decision, 'error')

    async def test_empty_and_oversized_input(self):
        self.assertEqual((await rails.guard_input(' ')).reason,'empty_input')
        self.assertEqual((await rails.guard_input('x'*6001)).reason,'input_too_long')

    async def test_native_nemo_decisions(self):
        for status, expected in [(RailStatus.PASSED,'allow'), (RailStatus.BLOCKED,'block'),
                                 (RailStatus.MODIFIED,'error')]:
            with self.subTest(status=status):
                result=Mock(status=status,content='refusal',rail='self check input')
                with patch.object(rails,'_rails',Mock(check_async=AsyncMock(return_value=result))):
                    outcome=await rails.guard_input('test question')
                self.assertEqual(outcome.decision,expected)

    async def test_provider_failure(self):
        with patch.object(rails,'_rails',Mock(check_async=AsyncMock(side_effect=TimeoutError()))):
            self.assertEqual((await rails.guard_input('question')).decision,'error')

    async def test_real_nemo_input_flow(self):
        """Run actual configured Colang input rails with simulated action verdicts."""
        from nemoguardrails import RailsConfig, LLMRails
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        from nemoguardrails.actions.rail_outcome import RailOutcome
        from app.Guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT
        config=RailsConfig.from_content(colang_content=COLANG_CONTENT,yaml_content=YAML_CONTENT)
        # Disable the optional heuristic detector in this controlled flow test.
        config.rails.input.flows=['self check input']
        engine=LLMRails(config,llm=FakeListChatModel(responses=['no']))
        for allowed in [True,False]:
            async def verdict(**kwargs):
                return RailOutcome.allow() if allowed else RailOutcome.block()
            engine.register_action(verdict,name='self_check_input')
            with patch.object(rails,'_rails',engine):
                result=await rails.guard_input('Explain Kubernetes')
            self.assertEqual(result.decision,'allow' if allowed else 'block')


if __name__=='__main__': unittest.main()
