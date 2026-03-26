from __future__ import annotations

from pathlib import Path
from typing import Literal

from taboo_arena.cards.schemas import CardRecord
from taboo_arena.config import RunSettings
from taboo_arena.engine.round_engine import RoundEngine
from taboo_arena.judge import GuessCanonicalizer, GuessMatchStatus, NormalizedLLMJudge
from taboo_arena.logging.run_logger import RunLogger
from taboo_arena.models.registry import ModelEntry
from taboo_arena.prompts.tasks import (
    build_clue_judge_messages,
    build_cluer_messages,
    build_guess_judge_messages,
    build_guesser_messages,
)
from tests.conftest import FakeModelManager

CLUE_ALLOW = (
    '{"allow":true,"block_reason_codes":[],"warnings":[],"matched_surface_forms":[],'
    '"judge_version":"clue_judge_v1"}'
)
CLUE_NEAR_EXPLICIT_ONLY = (
    '{"allow":false,"block_reason_codes":["near_explicit_paraphrase"],"warnings":[],'
    '"matched_surface_forms":["forest giant"],"judge_version":"clue_judge_v1"}'
)
GUESS_CORRECT = (
    '{"correct":true,"reason_codes":["exact_target_present"],"warnings":[],"matched_surface_forms":["Bear"],'
    '"judge_version":"guess_judge_v1"}'
)


def _entry(
    model_id: str,
    *,
    backend: Literal["transformers_safetensors", "llama_cpp_gguf"] = "transformers_safetensors",
) -> ModelEntry:
    return ModelEntry(
        id=model_id,
        display_name=model_id,
        backend=backend,
        repo_id=f"fake/{model_id}",
        architecture_family="fake",
        chat_template_id="generic_completion",
        supports_system_prompt=True,
        roles_supported=["cluer", "guesser", "judge"],
        languages=["en"],
        filename="model.gguf" if backend == "llama_cpp_gguf" else None,
    )


def _alias_card() -> CardRecord:
    return CardRecord(
        id="things:table:0007",
        target="Table",
        taboo_hard=["wood", "furniture", "chair"],
        aliases=["Desk"],
        source_category="things",
        source_repo="local/test",
        source_ref="main",
        category_label="Things",
    )


def test_active_gameplay_ignores_aliases_in_prompts_and_guess_matching() -> None:
    card = _alias_card()

    messages = [
        *build_cluer_messages(
            card=card,
            accepted_clues=[],
            rejected_clues=[],
            wrong_guesses=[],
            attempt_no=1,
            repair_no=1,
            allowed_angles=["type", "use", "context"],
            blocked_terms=["table"],
            blocked_prior_clues=[],
            blocked_angles=[],
        ),
        *build_clue_judge_messages(
            card=card,
            clue_draft="used for meals",
            accepted_clues=[],
            rejected_clues=[],
            attempt_no=1,
        ),
        *build_guesser_messages(
            card=card,
            current_clue="used for meals",
            accepted_clues=["used for meals"],
            wrong_guesses=[],
            attempt_no=1,
        ),
        *build_guess_judge_messages(
            card=card,
            guess_text="table",
            attempt_no=1,
            match_status="correct",
            match_reason="exact_match",
            candidate_spans=["table"],
            warnings=[],
        ),
    ]
    combined_prompt = "\n".join(message.content for message in messages).casefold()

    assert "desk" not in combined_prompt

    match = GuessCanonicalizer().match("desk", card.target)
    assert match.status is GuessMatchStatus.INCORRECT
    assert match.reason == "target_not_present"


def test_normalized_llm_judge_demotes_near_explicit_paraphrase_to_warning(
    sample_card,
) -> None:
    judge = NormalizedLLMJudge()
    manager = FakeModelManager(responses={"judge": [CLUE_NEAR_EXPLICIT_ONLY]})

    result, _ = judge.evaluate_clue(
        model_manager=manager,
        model_entry=_entry("judge"),
        card=sample_card,
        clue_draft="forest giant",
        accepted_clues=[],
        rejected_clues=[],
        attempt_no=1,
        generation_params=RunSettings().generation.judge,
        runtime_policy="keep_loaded_if_possible",
        device_preference="auto",
    )

    assert result.verdict == "pass"
    assert result.reasons == []
    assert result.warnings == ["near_explicit_paraphrase"]


def test_round_engine_treats_near_explicit_paraphrase_as_warning_only(
    sample_card,
    tmp_path: Path,
) -> None:
    logger = RunLogger(log_root=tmp_path, console_trace=False)
    manager = FakeModelManager(
        responses={
            "cluer": ['{"candidates":[{"angle":"type","clue":"forest giant"}]}'],
            "judge": [CLUE_NEAR_EXPLICIT_ONLY, GUESS_CORRECT],
            "guesser": ['{"guesses":["Bear","Wolf","Fox"]}'],
        }
    )
    engine = RoundEngine(
        model_manager=manager,
        logger=logger,
        settings=RunSettings(),
    )

    result = engine.play_round(
        card=sample_card,
        cluer_entry=_entry("cluer"),
        guesser_entry=_entry("guesser"),
        judge_entry=_entry("judge"),
    )

    assert result.solved is True
    clue_event = next(event for event in logger.events if event["event_type"] == "clue_accepted")
    assert clue_event["final_judge_verdict"] == "pass_with_warning"
    assert clue_event["judge_warning_flags"] == ["near_explicit_paraphrase"]


def test_round_engine_applies_same_text_level_clue_rules_for_transformers_and_gguf(
    sample_card,
    tmp_path: Path,
) -> None:
    cluer_output = (
        '{"candidates":['
        '{"angle":"type","clue":"Bear animal"},'
        '{"angle":"use","clue":"grizzly sleeper"},'
        '{"angle":"context","clue":"pooh friend"}'
        ']}'
    )
    settings = RunSettings(max_clue_repairs=1)

    tx_logger = RunLogger(log_root=tmp_path / "tx", console_trace=False)
    gguf_logger = RunLogger(log_root=tmp_path / "gguf", console_trace=False)
    tx_engine = RoundEngine(
        model_manager=FakeModelManager(responses={"tx-cluer": [cluer_output]}),
        logger=tx_logger,
        settings=settings,
    )
    gguf_engine = RoundEngine(
        model_manager=FakeModelManager(responses={"gguf-cluer": [cluer_output]}),
        logger=gguf_logger,
        settings=settings,
    )

    tx_result = tx_engine.play_round(
        card=sample_card,
        cluer_entry=_entry("tx-cluer", backend="transformers_safetensors"),
        guesser_entry=_entry("guesser"),
        judge_entry=_entry("judge"),
    )
    gguf_result = gguf_engine.play_round(
        card=sample_card,
        cluer_entry=_entry("gguf-cluer", backend="llama_cpp_gguf"),
        guesser_entry=_entry("guesser"),
        judge_entry=_entry("judge"),
    )

    assert tx_result.terminal_reason == "clue_not_repaired"
    assert gguf_result.terminal_reason == "clue_not_repaired"

    tx_validation_event = next(
        event for event in tx_logger.events if event["event_type"] == "clue_candidate_validation_completed"
    )
    gguf_validation_event = next(
        event
        for event in gguf_logger.events
        if event["event_type"] == "clue_candidate_validation_completed"
    )
    assert tx_validation_event["candidate_results"] == gguf_validation_event["candidate_results"]
