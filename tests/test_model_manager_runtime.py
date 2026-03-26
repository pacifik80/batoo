from __future__ import annotations

from pathlib import Path

from taboo_arena.config import GenerationParams
from taboo_arena.models.backends import GenerationResponse
from taboo_arena.models.manager import ModelManager
from taboo_arena.models.registry import ModelEntry
from taboo_arena.prompts import PromptMessage


class _FakeRuntimeDiagnostics:
    def transformers_gpu_ready(self) -> bool:
        return True

    def gguf_gpu_ready(self) -> bool:
        return False


class _FakeBackend:
    init_count = 0

    def __init__(
        self,
        *,
        model_path: str,
        tokenizer_repo: str | None,
        device_preference: str,
        runtime_policy: str,
    ) -> None:
        type(self).init_count += 1
        self.runtime_summary = (
            f"fake:{device_preference}:{runtime_policy}:{Path(model_path).name}:{tokenizer_repo or 'none'}"
        )

    def generate(
        self,
        *,
        prompt: str,
        generation_params: GenerationParams,
        stop_tokens: list[str],
        prompt_template_id: str,
        banned_phrases: list[str] | None = None,
    ) -> GenerationResponse:
        self.last_banned_phrases = list(banned_phrases or [])
        return GenerationResponse(
            text="ok",
            prompt_tokens=10,
            completion_tokens=2,
            latency_ms=1.0,
            prompt_template_id=prompt_template_id,
        )

    def supports_cpu_offload(self) -> bool:
        return False

    def activate_for_inference(self) -> None:
        return None

    def offload_to_cpu(self) -> None:
        return None


class _FakeOffloadBackend(_FakeBackend):
    activate_count = 0
    offload_count = 0

    def __init__(
        self,
        *,
        model_path: str,
        tokenizer_repo: str | None,
        device_preference: str,
        runtime_policy: str,
    ) -> None:
        super().__init__(
            model_path=model_path,
            tokenizer_repo=tokenizer_repo,
            device_preference=device_preference,
            runtime_policy=runtime_policy,
        )
        self.residency = "gpu"

    def supports_cpu_offload(self) -> bool:
        return True

    def activate_for_inference(self) -> None:
        type(self).activate_count += 1
        self.residency = "gpu"
        self.runtime_summary = "fake-offload residency=gpu"

    def offload_to_cpu(self) -> None:
        type(self).offload_count += 1
        self.residency = "cpu"
        self.runtime_summary = "fake-offload residency=cpu"


def _entry(model_id: str, repo_id: str) -> ModelEntry:
    return ModelEntry(
        id=model_id,
        display_name=model_id,
        backend="transformers_safetensors",
        repo_id=repo_id,
        architecture_family="fake",
        chat_template_id="generic_completion",
        supports_system_prompt=True,
        roles_supported=["cluer", "guesser", "judge"],
        languages=["en"],
        estimated_vram_gb=7.0,
    )


def test_sequential_policy_reuses_same_physical_model_across_role_entries(monkeypatch, tmp_path: Path) -> None:
    _FakeBackend.init_count = 0
    monkeypatch.setattr("taboo_arena.models.manager.TransformersGenerator", _FakeBackend)
    monkeypatch.setattr(
        "taboo_arena.models.manager.get_runtime_diagnostics",
        lambda: _FakeRuntimeDiagnostics(),
    )

    manager = ModelManager()
    monkeypatch.setattr(manager, "ensure_model_available", lambda entry: tmp_path / entry.id)
    messages = [PromptMessage(role="user", content="Say one clue.")]
    params = GenerationParams()

    manager.generate(
        model_entry=_entry("cluer-a", "shared/repo"),
        messages=messages,
        generation_params=params,
        runtime_policy="sequential_load_unload",
        device_preference="auto",
        trace_role="cluer",
    )
    manager.generate(
        model_entry=_entry("judge-a", "shared/repo"),
        messages=messages,
        generation_params=params,
        runtime_policy="sequential_load_unload",
        device_preference="auto",
        trace_role="judge",
    )

    assert _FakeBackend.init_count == 1
    assert len(manager.loaded_models) == 1


def test_memory_assessment_dedupes_shared_model_across_roles(monkeypatch) -> None:
    monkeypatch.setattr(
        "taboo_arena.models.manager.get_runtime_diagnostics",
        lambda: _FakeRuntimeDiagnostics(),
    )

    manager = ModelManager()
    assessment = manager.assess_memory(
        [
            _entry("cluer-a", "shared/repo"),
            _entry("judge-a", "shared/repo"),
            _entry("guesser-b", "other/repo"),
        ],
        "keep_loaded_if_possible",
        device_preference="auto",
    )

    assert assessment.estimated_vram_gb == 14.0


def test_memory_assessment_can_recommend_cpu_offload(monkeypatch) -> None:
    monkeypatch.setattr(
        "taboo_arena.models.manager.get_runtime_diagnostics",
        lambda: _FakeRuntimeDiagnostics(),
    )
    monkeypatch.setattr(
        "taboo_arena.models.manager.get_memory_snapshot",
        lambda: type(
            "Snapshot",
            (),
            {
                "ram_total_gb": 64.0,
                "ram_available_gb": 40.0,
                "vram_total_gb": 12.0,
                "vram_available_gb": 10.0,
            },
        )(),
    )

    manager = ModelManager()
    assessment = manager.assess_memory(
        [
            _entry("cluer-a", "shared/repo-a"),
            _entry("judge-a", "shared/repo-b"),
            _entry("guesser-b", "shared/repo-c"),
        ],
        "keep_loaded_if_possible",
        device_preference="auto",
    )

    assert assessment.recommended_policy == "keep_cpu_offloaded_if_possible"
    assert assessment.peak_active_vram_gb == 7.0


def test_cpu_offload_policy_keeps_models_loaded_but_moves_inactive_ones_to_cpu(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _FakeOffloadBackend.init_count = 0
    _FakeOffloadBackend.activate_count = 0
    _FakeOffloadBackend.offload_count = 0
    monkeypatch.setattr("taboo_arena.models.manager.TransformersGenerator", _FakeOffloadBackend)
    monkeypatch.setattr(
        "taboo_arena.models.manager.get_runtime_diagnostics",
        lambda: _FakeRuntimeDiagnostics(),
    )

    manager = ModelManager()
    monkeypatch.setattr(manager, "ensure_model_available", lambda entry: tmp_path / entry.id)
    messages = [PromptMessage(role="user", content="Say one clue.")]
    params = GenerationParams()

    manager.generate(
        model_entry=_entry("cluer-a", "repo-a"),
        messages=messages,
        generation_params=params,
        runtime_policy="keep_cpu_offloaded_if_possible",
        device_preference="auto",
        trace_role="cluer",
    )
    manager.generate(
        model_entry=_entry("judge-a", "repo-b"),
        messages=messages,
        generation_params=params,
        runtime_policy="keep_cpu_offloaded_if_possible",
        device_preference="auto",
        trace_role="judge",
    )

    assert _FakeOffloadBackend.init_count == 2
    assert _FakeOffloadBackend.offload_count == 1
    assert _FakeOffloadBackend.activate_count == 2
    assert len(manager.loaded_models) == 2


def test_generate_forwards_banned_phrases_to_backend(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("taboo_arena.models.manager.TransformersGenerator", _FakeBackend)
    monkeypatch.setattr(
        "taboo_arena.models.manager.get_runtime_diagnostics",
        lambda: _FakeRuntimeDiagnostics(),
    )

    manager = ModelManager()
    monkeypatch.setattr(manager, "ensure_model_available", lambda entry: tmp_path / entry.id)
    messages = [PromptMessage(role="user", content="Say one clue.")]
    params = GenerationParams()

    manager.generate(
        model_entry=_entry("cluer-a", "shared/repo"),
        messages=messages,
        generation_params=params,
        runtime_policy="keep_loaded_if_possible",
        device_preference="auto",
        trace_role="cluer",
        banned_phrases=["Indiana Jones", "actor"],
    )

    backend = next(iter(manager.loaded_models.values()))
    assert isinstance(backend, _FakeBackend)
    assert backend.last_banned_phrases == ["Indiana Jones", "actor"]
