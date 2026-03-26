from __future__ import annotations

from taboo_arena.judge.logical import LogicalValidator
from taboo_arena.models.constraint_compiler import ConstraintCompiler
from taboo_arena.models.registry import ModelEntry


class _FakeTokenizer:
    eos_token_id = 999999

    def __init__(self) -> None:
        self.seen_inputs: list[str] = []

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        self.seen_inputs.append(text)
        return [ord(char) for char in text]


class _FakeTransformersBackend:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()


class _FakeGGUFLlm:
    def __init__(self) -> None:
        self.seen_inputs: list[bytes] = []

    def tokenize(
        self,
        text: bytes,
        *,
        add_bos: bool = False,
        special: bool = False,
    ) -> list[int]:
        self.seen_inputs.append(text)
        return [int(byte) for byte in text]


class _FakeGGUFBackend:
    def __init__(self) -> None:
        self.llm = _FakeGGUFLlm()


def _transformers_entry(model_id: str) -> ModelEntry:
    return ModelEntry(
        id=model_id,
        display_name=model_id,
        backend="transformers_safetensors",
        repo_id=f"fake/{model_id}",
        architecture_family="fake",
        chat_template_id="generic_completion",
        supports_system_prompt=True,
        roles_supported=["cluer", "guesser", "judge"],
        languages=["en"],
    )


def _gguf_entry(model_id: str) -> ModelEntry:
    return ModelEntry(
        id=model_id,
        display_name=model_id,
        backend="llama_cpp_gguf",
        repo_id=f"fake/{model_id}",
        filename="model.gguf",
        architecture_family="fake",
        chat_template_id="generic_completion",
        supports_system_prompt=True,
        roles_supported=["cluer", "guesser", "judge"],
        languages=["en"],
    )


def test_constraint_compiler_builds_transformers_token_constraints_per_role_model() -> None:
    compiler = ConstraintCompiler()
    backend = _FakeTransformersBackend()

    compiled = compiler.compile(
        role="cluer",
        model_entry=_transformers_entry("cluer-a"),
        backend=backend,
        forbidden_surface_forms=["Bear", "grizzly", "Bear"],
        json_output_required=True,
    )

    assert compiled.role == "cluer"
    assert compiled.model_id == "cluer-a"
    assert compiled.backend_name == "transformers_safetensors"
    assert compiled.forbidden_surface_forms == ["Bear", "grizzly"]
    assert compiled.supports_decode_enforcement is True
    assert compiled.transformers_bad_words_ids
    assert compiled.json_prefix_token_sequences
    assert compiled.gguf_banned_token_sequences == []
    assert "Bear" in backend.tokenizer.seen_inputs or " bear" in backend.tokenizer.seen_inputs
    assert "{" in backend.tokenizer.seen_inputs


def test_constraint_compiler_builds_gguf_token_constraints_via_llama_tokenization() -> None:
    compiler = ConstraintCompiler()
    backend = _FakeGGUFBackend()

    compiled = compiler.compile(
        role="guesser",
        model_entry=_gguf_entry("guesser-a"),
        backend=backend,
        forbidden_surface_forms=["wolf"],
        json_output_required=True,
    )

    assert compiled.role == "guesser"
    assert compiled.model_id == "guesser-a"
    assert compiled.backend_name == "llama_cpp_gguf"
    assert compiled.forbidden_surface_forms == ["wolf"]
    assert compiled.supports_decode_enforcement is False
    assert compiled.transformers_bad_words_ids == []
    assert compiled.gguf_banned_token_sequences
    assert compiled.json_prefix_token_sequences
    assert b"wolf" in backend.llm.seen_inputs or b" wolf" in backend.llm.seen_inputs
    assert b"{" in backend.llm.seen_inputs


def test_text_level_validator_is_backend_agnostic(sample_card) -> None:
    validator = LogicalValidator()

    transformers_result = validator.validate(
        "Bear cave",
        card=sample_card,
        previous_accepted_clues=[],
    )
    gguf_result = validator.validate(
        "Bear cave",
        card=sample_card,
        previous_accepted_clues=[],
    )

    assert transformers_result.model_dump(mode="json") == gguf_result.model_dump(mode="json")
