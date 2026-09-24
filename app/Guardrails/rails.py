import os
import logging
from dataclasses import dataclass
from typing import Literal

import logfire

from langchain_groq import ChatGroq
from nemoguardrails import RailsConfig, LLMRails
from nemoguardrails.rails.llm.options import RailStatus, RailType

from app.config import settings
from app.Guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT


@dataclass(frozen=True)
class GuardResult:
    decision: Literal["allow", "block", "error"]
    reason: str
    response: str | None = None


_logger = logging.getLogger(__name__)

_rails: LLMRails | None = None


GUARD_UNAVAILABLE = GuardResult(
    decision="error",
    reason="guard_unavailable",
    response="The assistant is temporarily unavailable. Please try again.",
)


def initialize_rails() -> None:
    """Initialize NeMo Guardrails once when the application starts."""

    global _rails

    guard_llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=os.getenv("GUARDRAIL_MODEL", "openai/gpt-oss-20b"),
        temperature=0,
        timeout=20,
        max_retries=0,
    )

    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=YAML_CONTENT,
    )

    _rails = LLMRails(
        config,
        llm=guard_llm,
    )

    logfire.info("NeMo Guardrails initialized")


async def guard_input(message: str) -> GuardResult:
    """Run only the NeMo input rails."""

    if not isinstance(message, str) or not message.strip():
        return GuardResult(
            "block",
            "empty_input",
            "Please enter a question.",
        )

    if len(message) > 6000:
        return GuardResult(
            "block",
            "input_too_long",
            "Please shorten your question.",
        )

    if _rails is None:
        _logger.error("Guardrails were not initialized")
        return GUARD_UNAVAILABLE

    try:
        result = await _rails.check_async(
            messages=[
                {
                    "role": "user",
                    "content": message,
                }
            ],
            rail_types=[RailType.INPUT],
        )

    except Exception as exc:
        _logger.warning(
            "Guardrail check failed: %s",
            type(exc).__name__,
        )
        return GUARD_UNAVAILABLE

    if result.status == RailStatus.BLOCKED:
        logfire.info(
            "Guardrail blocked input: rail={rail}",
            rail=result.rail,
        )

        return GuardResult(
            decision="block",
            reason=str(result.rail or "input_policy"),
            response=(
                result.content
                or "I can't help with that request."
            ),
        )

    if result.status == RailStatus.MODIFIED:
        # You are not currently using modifying rails,
        # but handle the state explicitly.
        return GuardResult(
            decision="error",
            reason="unexpected_modified_input",
            response=GUARD_UNAVAILABLE.response,
        )

    logfire.info("Guardrail input passed")

    return GuardResult(
        decision="allow",
        reason="allowed",
    )