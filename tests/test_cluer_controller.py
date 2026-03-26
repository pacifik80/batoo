from __future__ import annotations

from taboo_arena.engine.cluer_controller import (
    ClueAngle,
    build_repair_feedback,
    evaluate_clue_candidates,
    parse_cluer_candidates,
    select_allowed_angles,
    select_best_candidate,
)
from taboo_arena.judge.logical import LogicalValidator


def test_parse_cluer_candidates_reads_structured_json() -> None:
    candidates, parse_mode = parse_cluer_candidates(
        '{"candidates":[{"angle":"type","clue":"large forest mammal"},{"angle":"use","clue":"found in the woods"}]}',
        allowed_angles=[ClueAngle.TYPE, ClueAngle.USE, ClueAngle.CONTEXT],
    )

    assert parse_mode == "json"
    assert [candidate.angle for candidate in candidates] == ["type", "use"]
    assert [candidate.clue for candidate in candidates] == ["large forest mammal", "found in the woods"]


def test_select_allowed_angles_rotates_and_avoids_used_angles() -> None:
    first = select_allowed_angles(attempt_no=1, used_angles=[], blocked_angles=[])
    second = select_allowed_angles(
        attempt_no=2,
        used_angles=[first[0].value],
        blocked_angles=[],
    )

    assert len(first) == 3
    assert len(second) == 3
    assert second[0] is not first[0]
    assert first[0].value not in [angle.value for angle in second]


def test_evaluate_clue_candidates_filters_and_selects_best(sample_card) -> None:
    validator = LogicalValidator()
    candidates, _ = parse_cluer_candidates(
        '{"candidates":[{"angle":"type","clue":"Bear animal"},{"angle":"use","clue":"hibernates through winter"},{"angle":"context","clue":"forest giant"}]}',
        allowed_angles=[ClueAngle.TYPE, ClueAngle.USE, ClueAngle.CONTEXT],
    )

    evaluations = evaluate_clue_candidates(
        candidates=candidates,
        validator=validator,
        card=sample_card,
        previous_accepted_clues=[],
        previous_rejected_clues=[],
        used_angles=[],
    )
    selected = select_best_candidate(evaluations)

    assert [item.logical_result.verdict for item in evaluations] == ["fail", "pass", "pass"]
    assert selected is not None
    assert selected.angle is ClueAngle.USE


def test_build_repair_feedback_returns_machine_friendly_codes(sample_card) -> None:
    validator = LogicalValidator()
    candidates, _ = parse_cluer_candidates(
        '{"candidates":[{"angle":"type","clue":"Bear animal"},{"angle":"effect","clue":"grizzly hibernator"}]}',
        allowed_angles=[ClueAngle.TYPE, ClueAngle.EFFECT, ClueAngle.CONTEXT],
    )
    evaluations = evaluate_clue_candidates(
        candidates=candidates,
        validator=validator,
        card=sample_card,
        previous_accepted_clues=[],
        previous_rejected_clues=[],
        used_angles=[],
    )

    feedback = build_repair_feedback(
        evaluations=evaluations,
        allowed_angles=[ClueAngle.TYPE, ClueAngle.EFFECT, ClueAngle.CONTEXT],
        blocked_angles=[ClueAngle.TYPE, ClueAngle.EFFECT],
    )

    assert "token_match" in feedback.reason_codes or "whole_word_match" in feedback.reason_codes
    assert "Bear" in feedback.blocked_terms or "grizzly" in feedback.blocked_terms
    assert feedback.blocked_angles == ["type", "effect"]
