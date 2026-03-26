"""Merge deterministic and LLM judge results."""

from __future__ import annotations

from pydantic import BaseModel, Field

from taboo_arena.judge.llm import LLMJudgeResult
from taboo_arena.judge.logical import LogicalValidationResult


class MergedJudgeResult(BaseModel):
    """Merged judge result used by the engine."""

    final_verdict: str
    judge_disagreement: bool
    logical_verdict: str
    logical_violations: list[str] = Field(default_factory=list)
    llm_judge_verdict: str
    llm_judge_reasons: list[str] = Field(default_factory=list)


def merge_judge_results(
    logical: LogicalValidationResult,
    llm: LLMJudgeResult,
    *,
    block_on_uncertain: bool,
) -> MergedJudgeResult:
    """Merge the two judge layers according to the configured policy."""
    if logical.verdict == "fail":
        final = "fail"
    elif llm.verdict == "fail":
        final = "fail"
    elif llm.verdict == "uncertain":
        final = "fail" if block_on_uncertain else "pass_with_warning"
    else:
        final = "pass"

    disagreement = logical.verdict != ("fail" if llm.verdict == "fail" else "pass")
    return MergedJudgeResult(
        final_verdict=final,
        judge_disagreement=disagreement,
        logical_verdict=logical.verdict,
        logical_violations=logical.violations,
        llm_judge_verdict=llm.verdict,
        llm_judge_reasons=llm.reasons,
    )

