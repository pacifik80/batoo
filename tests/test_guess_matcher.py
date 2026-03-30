from __future__ import annotations

from taboo_arena.judge.guess_matcher import GuessCanonicalizer, GuessMatchStatus


def test_guess_canonicalizer_matches_refactor_examples() -> None:
    matcher = GuessCanonicalizer()

    exact = matcher.match("table", "table")
    wrapped = matcher.match("the table", "table")
    boilerplate = matcher.match("I think it is table", "table")
    multi = matcher.match("table or desk", "table")
    wrong = matcher.match("desk", "table")

    assert exact.status is GuessMatchStatus.CORRECT
    assert exact.reason == "exact_match"
    assert wrapped.status is GuessMatchStatus.CORRECT
    assert wrapped.reason == "normalized_wrapper_match"
    assert boilerplate.status is GuessMatchStatus.CORRECT
    assert boilerplate.reason == "normalized_wrapper_match"
    assert multi.status is GuessMatchStatus.CORRECT_MULTI_CANDIDATE
    assert multi.reason == "multi_candidate_contains_target"
    assert wrong.status is GuessMatchStatus.INCORRECT
    assert wrong.reason == "target_not_present"


def test_guess_shortlist_filters_wrapper_repeats_but_keeps_new_target_candidate() -> None:
    matcher = GuessCanonicalizer()
    evaluations = matcher.evaluate_shortlist(
        ["my guess is wolf", "the wolf", "table or wolf"],
        target="table",
        previous_wrong_guesses=["wolf"],
    )

    assert evaluations[0].invalid_reason == "repeat_previous_wrong_guess"
    assert evaluations[1].invalid_reason == "repeat_previous_wrong_guess"
    assert evaluations[2].is_valid_visible_candidate is True
    assert evaluations[2].match_result.status is GuessMatchStatus.CORRECT_MULTI_CANDIDATE


def test_guess_shortlist_rejects_structured_payload_candidates() -> None:
    matcher = GuessCanonicalizer()
    evaluations = matcher.evaluate_shortlist(
        ['{"guesses":["table","desk","chair"]}', "table"],
        target="table",
        previous_wrong_guesses=[],
    )

    assert evaluations[0].invalid_reason == "structured_payload_guess"
    assert evaluations[0].match_result.reason == "structured_payload_detected"
    assert evaluations[1].is_valid_visible_candidate is True
