from __future__ import annotations

from taboo_arena.models.registry import ModelEntry
from taboo_arena.prompts.tasks import build_cluer_messages
from taboo_arena.prompts.templates import PromptMessage, render_prompt


def _entry(profile_id: str = "compact_small") -> ModelEntry:
    return ModelEntry(
        id=f"test-{profile_id}",
        display_name=f"test-{profile_id}",
        backend="transformers_safetensors",
        repo_id="test/test",
        architecture_family="test",
        chat_template_id="generic_completion",
        prompt_profile_id=profile_id,
        supports_system_prompt=True,
        roles_supported=["cluer", "guesser", "judge"],
        languages=["en"],
    )


def test_qwen_render_works_with_layered_prompt_content(sample_card) -> None:
    messages = build_cluer_messages(
        card=sample_card,
        accepted_clues=[],
        rejected_clues=[],
        wrong_guesses=[],
        attempt_no=1,
        repair_no=1,
        allowed_angles=["type", "use", "context"],
        blocked_terms=["bear"],
        blocked_prior_clues=[],
        blocked_angles=[],
        model_entry=_entry("compact_small"),
    )

    rendered = render_prompt(
        "qwen_chatml",
        messages,
        supports_system_prompt=True,
        stop_tokens=[],
    )

    assert "You are Qwen, created by Alibaba Cloud." in rendered.prompt
    assert "You are the Cluer in a local Taboo benchmark." in rendered.prompt
    assert "CONTROL_STATE" in rendered.prompt
    assert rendered.prompt.endswith("<|im_start|>assistant\n")


def test_mistral_render_preserves_single_integrated_instruction(sample_card) -> None:
    messages = build_cluer_messages(
        card=sample_card,
        accepted_clues=[],
        rejected_clues=[],
        wrong_guesses=[],
        attempt_no=1,
        repair_no=1,
        allowed_angles=["type", "use", "context"],
        blocked_terms=["bear"],
        blocked_prior_clues=[],
        blocked_angles=[],
        model_entry=_entry("standard"),
    )

    rendered = render_prompt(
        "mistral_inst",
        messages,
        supports_system_prompt=True,
        stop_tokens=[],
    )

    assert "You are the Cluer in a local Taboo benchmark." in rendered.prompt
    assert "Current state" in rendered.prompt
    assert 'target: Bear' in rendered.prompt
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


def test_other_template_families_render_smoke() -> None:
    llama = render_prompt(
        "llama3_chat",
        [PromptMessage(role="user", content="hello")],
        supports_system_prompt=True,
        stop_tokens=[],
    )
    gemma = render_prompt(
        "gemma_chat",
        [PromptMessage(role="user", content="hello")],
        supports_system_prompt=True,
        stop_tokens=[],
    )

    assert "<|begin_of_text|>" in llama.prompt
    assert "<|start_header_id|>user<|end_header_id|>" in llama.prompt
    assert "<bos>" in gemma.prompt
    assert "<start_of_turn>user" in gemma.prompt


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
