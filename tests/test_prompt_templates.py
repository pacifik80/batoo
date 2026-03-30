from __future__ import annotations

from taboo_arena.prompts.store import render_prompt_messages
from taboo_arena.prompts.templates import PromptMessage, render_prompt


def test_mistral_render_preserves_single_integrated_instruction() -> None:
    messages = render_prompt_messages(
        "cluer_candidates",
        {
            "target": "Bear",
            "forbidden_words_json": "[\"grizzly\", \"honey\", \"pooh\"]",
            "allowed_angles_json": "[\"type\", \"use\", \"context\"]",
            "blocked_terms_json": "[\"bear\"]",
            "blocked_prior_clues_json": "[\"forest mammal\"]",
            "blocked_angles_json": "[\"effect\"]",
            "previous_accepted_clues_json": "[\"forest mammal\"]",
            "previous_rejected_clues_json": "[\"cave dweller\"]",
            "previous_wrong_guesses_json": "[\"wolf\"]",
            "attempt_no": 2,
            "repair_no": 1,
            "repair_feedback_json": "{}",
            "output_schema_json": "{\"candidates\":[{\"angle\":\"type\",\"clue\":\"string\"}]}",
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
    assert "Allowed angles" in rendered.prompt
    assert rendered.prompt != "[INST] [/INST]"


def test_phi_fallback_render_includes_end_delimiters() -> None:
    rendered = render_prompt(
        "phi_chat",
        [
            PromptMessage(role="user", content="hello"),
        ],
        supports_system_prompt=True,
        stop_tokens=[],
    )

    assert rendered.prompt == "<|user|>hello<|end|><|assistant|>"
    assert rendered.add_special_tokens is True


def test_render_prompt_prefers_tokenizer_chat_template_when_available() -> None:
    calls: list[object] = []

    class FakeTokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            calls.append(messages)
            assert tokenize is False
            assert add_generation_prompt is True
            return "<tokenizer-template>"

    rendered = render_prompt(
        "phi_chat",
        [
            PromptMessage(role="user", content="hello"),
        ],
        supports_system_prompt=True,
        stop_tokens=["<|end|>"],
        tokenizer=FakeTokenizer(),
    )

    assert calls == [[{"role": "user", "content": "hello"}]]
    assert rendered.prompt == "<tokenizer-template>"
    assert rendered.add_special_tokens is False
