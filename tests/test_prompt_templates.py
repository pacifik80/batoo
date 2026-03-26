from __future__ import annotations

from taboo_arena.prompts.store import render_prompt_messages
from taboo_arena.prompts.templates import render_prompt


def test_mistral_render_preserves_single_integrated_instruction() -> None:
    messages = render_prompt_messages(
        "cluer",
        {
            "target": "Bear",
            "forbidden_words_csv": "grizzly, honey, pooh",
            "aliases_csv": "Grizzly Bear",
            "previous_accepted_clues_json": "[\"forest mammal\"]",
            "previous_rejected_clues_json": "[\"cave dweller\"]",
            "previous_wrong_guesses_json": "[\"wolf\"]",
            "attempt_no": 2,
            "repair_no": 1,
            "last_rejection_feedback": "Rules failed: token_match.",
        },
    )

    rendered = render_prompt(
        "mistral_inst",
        messages,
        supports_system_prompt=True,
        stop_tokens=[],
    )

    assert "You are the Cluer in a local Taboo benchmark." in rendered.prompt
    assert "Target: Bear." in rendered.prompt
    assert rendered.prompt != "[INST] [/INST]"
