"""Model download, inspection, loading, and runtime policy management."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download, snapshot_download, try_to_load_from_cache
from huggingface_hub.errors import GatedRepoError

from taboo_arena.config import DevicePreference, GenerationParams, MemoryPolicy
from taboo_arena.models.backends import (
    BackendRuntimeCapabilities,
    GenerationResponse,
    LlamaCppGenerator,
    ModelRuntimeError,
    TextGenerationBackend,
    TransformersGenerator,
    unload_backend_resources,
)
from taboo_arena.models.registry import ModelEntry
from taboo_arena.prompts import PromptMessage, render_prompt
from taboo_arena.utils.system import (
    SystemMemorySnapshot,
    get_memory_snapshot,
    get_runtime_diagnostics,
)


class ModelAuthError(RuntimeError):
    """Raised when Hugging Face authentication is required."""


@dataclass(slots=True)
class MemoryAssessment:
    """Estimated memory fit for the currently selected role models."""

    requested_policy: MemoryPolicy
    recommended_policy: MemoryPolicy
    estimated_vram_gb: float
    peak_active_vram_gb: float
    snapshot: SystemMemorySnapshot
    fits_requested_policy: bool
    reason: str


class ModelManager:
    """Inspect, download, load, unload, and generate with local role models."""

    def __init__(self, logger: Any | None = None) -> None:
        self.logger = logger
        self.loaded_models: dict[str, TextGenerationBackend] = {}
        self._active_runtime_policy: MemoryPolicy = "keep_loaded_if_possible"

    def inspect_cache(self, entry: ModelEntry) -> bool:
        """Return whether the selected model already exists in the HF cache."""
        try:
            if entry.backend == "llama_cpp_gguf":
                if entry.filename is None:
                    raise ModelRuntimeError("GGUF entry is missing a filename.")
                cached = try_to_load_from_cache(
                    repo_id=entry.repo_id,
                    filename=entry.filename,
                    revision=entry.revision,
                )
                return cached is not None
            cached = try_to_load_from_cache(
                repo_id=entry.repo_id,
                filename="config.json",
                revision=entry.revision,
            )
            return cached is not None
        except Exception:
            return False

    def is_loaded(
        self,
        entry: ModelEntry,
        *,
        device_preference: DevicePreference = "auto",
    ) -> bool:
        """Return whether the selected model currently resides in the active process."""
        runtime_key = self._model_runtime_key(
            entry,
            device_preference=device_preference,
        )
        return runtime_key in self.loaded_models

    def ensure_model_available(self, entry: ModelEntry) -> Path:
        """Ensure the selected model exists locally, downloading it if necessary."""
        token = os.getenv("HF_TOKEN")
        if entry.requires_hf_auth and not token:
            raise ModelAuthError(
                f"Model '{entry.display_name}' requires Hugging Face authentication. Set HF_TOKEN."
            )

        try:
            if entry.backend == "llama_cpp_gguf":
                if entry.filename is None:
                    raise ModelRuntimeError("GGUF entry is missing a filename.")
                if self.logger is not None:
                    self.logger.emit(
                        "model_download_started",
                        model_id=entry.id,
                        state="downloading_model",
                        backend=entry.backend,
                    )
                path = hf_hub_download(
                    repo_id=entry.repo_id,
                    filename=entry.filename,
                    revision=entry.revision,
                    token=token or None,
                )
            else:
                if self.logger is not None:
                    self.logger.emit(
                        "model_download_started",
                        model_id=entry.id,
                        state="downloading_model",
                        backend=entry.backend,
                    )
                path = snapshot_download(
                    repo_id=entry.repo_id,
                    revision=entry.revision,
                    token=token or None,
                    allow_patterns=[
                        "*.py",
                        "**/*.py",
                        "*.json",
                        "*.model",
                        "*.safetensors",
                        "*.txt",
                        "*.tiktoken",
                        "tokenizer*",
                        "special_tokens_map.json",
                        "generation_config.json",
                    ],
                )
        except GatedRepoError as exc:
            raise ModelAuthError(
                f"Model '{entry.display_name}' is gated on Hugging Face. Check access and HF_TOKEN."
            ) from exc
        except Exception as exc:
            raise ModelRuntimeError(f"Failed to download model '{entry.display_name}': {exc}") from exc

        if self.logger is not None:
            self.logger.emit(
                "model_download_finished",
                model_id=entry.id,
                state="idle",
                backend=entry.backend,
            )
        return Path(path)

    def load_model(
        self,
        entry: ModelEntry,
        *,
        device_preference: DevicePreference,
        runtime_key: str | None = None,
    ) -> TextGenerationBackend:
        """Load a model backend into memory."""
        resolved_runtime_key = runtime_key or self._model_runtime_key(
            entry,
            device_preference=device_preference,
        )
        if resolved_runtime_key in self.loaded_models:
            backend = self.loaded_models[resolved_runtime_key]
            if self.logger is not None:
                self.logger.emit(
                    "model_reused",
                    model_id=entry.id,
                    backend=entry.backend,
                    runtime_key=resolved_runtime_key,
                    runtime_summary=getattr(backend, "runtime_summary", "unknown"),
                    state="idle",
                )
            return backend

        if self.logger is not None:
            self.logger.emit(
                "model_load_started",
                model_id=entry.id,
                backend=entry.backend,
                runtime_key=resolved_runtime_key,
                state="loading_model",
            )
        model_path = self.ensure_model_available(entry)
        if entry.backend == "transformers_safetensors":
            backend = TransformersGenerator(
                model_path=str(model_path),
                tokenizer_repo=entry.tokenizer_repo,
                device_preference=device_preference,
                runtime_policy=self._active_runtime_policy,
            )
        else:
            gguf_path = model_path if model_path.is_file() else model_path / str(entry.filename)
            backend = LlamaCppGenerator(
                model_path=str(gguf_path),
                device_preference=device_preference,
            )
        self.loaded_models[resolved_runtime_key] = backend
        if self.logger is not None:
            self.logger.emit(
                "model_load_finished",
                model_id=entry.id,
                backend=entry.backend,
                runtime_key=resolved_runtime_key,
                runtime_summary=getattr(backend, "runtime_summary", "unknown"),
                state="idle",
            )
        return backend

    def unload_runtime_key(self, runtime_key: str) -> None:
        """Unload one loaded backend by runtime key."""
        if self.loaded_models.pop(runtime_key, None) is not None:
            unload_backend_resources()

    def unload_all(self, *, keep_runtime_key: str | None = None) -> None:
        """Unload all loaded models except the optional one to keep."""
        removable = [runtime_key for runtime_key in self.loaded_models if runtime_key != keep_runtime_key]
        for runtime_key in removable:
            self.loaded_models.pop(runtime_key, None)
        if removable:
            unload_backend_resources()

    def generate(
        self,
        *,
        model_entry: ModelEntry,
        messages: list[PromptMessage],
        generation_params: GenerationParams,
        runtime_policy: MemoryPolicy,
        device_preference: DevicePreference,
        trace_role: str = "unknown",
        banned_phrases: list[str] | None = None,
    ) -> GenerationResponse:
        """Render a registry-driven prompt and generate text."""
        self._active_runtime_policy = runtime_policy
        runtime_key = self._model_runtime_key(
            model_entry,
            device_preference=device_preference,
        )
        if runtime_policy == "sequential_load_unload":
            self.unload_all(keep_runtime_key=runtime_key)
        elif runtime_policy == "keep_cpu_offloaded_if_possible":
            self._prepare_cpu_offload_switch(keep_runtime_key=runtime_key)
        backend = self.load_model(
            model_entry,
            device_preference=device_preference,
            runtime_key=runtime_key,
        )
        if runtime_policy == "keep_cpu_offloaded_if_possible":
            self._activate_backend(runtime_key, backend)
        rendered = render_prompt(
            model_entry.chat_template_id,
            messages,
            supports_system_prompt=model_entry.supports_system_prompt,
            stop_tokens=model_entry.stop_tokens,
        )
        if self.logger is not None and hasattr(self.logger, "trace_prompt"):
            self.logger.trace_prompt(
                role=trace_role,
                model_id=model_entry.id,
                prompt_template_id=rendered.prompt_template_id,
                prompt=rendered.prompt,
                generation_params=generation_params.model_dump(mode="json"),
            )
        response = backend.generate(
            prompt=rendered.prompt,
            generation_params=generation_params,
            stop_tokens=rendered.stop_tokens,
            prompt_template_id=rendered.prompt_template_id,
            banned_phrases=banned_phrases,
        )
        if self.logger is not None and hasattr(self.logger, "trace_response"):
            self.logger.trace_response(
                role=trace_role,
                model_id=model_entry.id,
                text=response.text,
                latency_ms=response.latency_ms,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
            )
        return response

    def runtime_capabilities(
        self,
        entry: ModelEntry,
        *,
        device_preference: DevicePreference = "auto",
        runtime_policy: MemoryPolicy = "keep_loaded_if_possible",
    ) -> BackendRuntimeCapabilities:
        """Expose explicit runtime capabilities without changing generation behavior."""
        if entry.backend == "transformers_safetensors":
            supports_cpu_offload = bool(
                device_preference != "cpu"
                and runtime_policy == "keep_cpu_offloaded_if_possible"
            )
            return BackendRuntimeCapabilities(
                backend_name=entry.backend,
                supports_banned_phrase_enforcement=True,
                supports_cpu_offload=supports_cpu_offload,
            )
        return BackendRuntimeCapabilities(
            backend_name=entry.backend,
            supports_banned_phrase_enforcement=False,
            supports_cpu_offload=False,
        )

    def assess_memory(
        self,
        entries: list[ModelEntry],
        requested_policy: MemoryPolicy,
        *,
        device_preference: DevicePreference = "auto",
    ) -> MemoryAssessment:
        """Estimate whether selected models fit in memory."""
        unique_entries: dict[str, ModelEntry] = {}
        for entry in entries:
            unique_entries.setdefault(
                self._model_runtime_key(entry, device_preference=device_preference),
                entry,
            )
        estimated = sum(entry.estimated_vram_gb or 0.0 for entry in unique_entries.values())
        peak_active = max((entry.estimated_vram_gb or 0.0 for entry in unique_entries.values()), default=0.0)
        snapshot = get_memory_snapshot()
        available_vram = snapshot.vram_available_gb
        fits_keep_loaded = estimated <= available_vram * 0.85 if available_vram else False
        fits_cpu_offload = self._can_keep_cpu_offloaded(
            list(unique_entries.values()),
            peak_active_vram_gb=peak_active,
            total_estimated_vram_gb=estimated,
            snapshot=snapshot,
            device_preference=device_preference,
        )
        recommended = requested_policy
        reason = "Requested policy accepted."
        if requested_policy == "keep_loaded_if_possible":
            if fits_keep_loaded:
                reason = "Estimated memory use fits available VRAM."
            elif fits_cpu_offload:
                recommended = "keep_cpu_offloaded_if_possible"
                reason = "Models do not all fit in VRAM, but they can stay loaded with inactive transformers offloaded to CPU RAM."
            else:
                recommended = "sequential_load_unload"
                reason = "Estimated memory use exceeds available resources, falling back to sequential loading."
        elif requested_policy == "keep_cpu_offloaded_if_possible":
            if fits_cpu_offload:
                reason = "Estimated peak VRAM and total RAM fit CPU offload mode."
            elif fits_keep_loaded:
                recommended = "keep_loaded_if_possible"
                reason = "All models already fit in VRAM, so full keep-loaded mode is better."
            else:
                recommended = "sequential_load_unload"
                reason = "CPU offload mode is not viable for this selection, falling back to sequential loading."
        else:
            reason = "Estimated memory use fits available resources."
        return MemoryAssessment(
            requested_policy=requested_policy,
            recommended_policy=recommended,
            estimated_vram_gb=round(estimated, 2),
            peak_active_vram_gb=round(peak_active, 2),
            snapshot=snapshot,
            fits_requested_policy=requested_policy == recommended,
            reason=reason,
        )

    def resolve_runtime_policy(
        self,
        entries: list[ModelEntry],
        *,
        requested_policy: MemoryPolicy,
        device_preference: DevicePreference = "auto",
    ) -> MemoryPolicy:
        """Choose the runtime loading policy for a set of models."""
        assessment = self.assess_memory(
            entries,
            requested_policy,
            device_preference=device_preference,
        )
        return assessment.recommended_policy

    def _model_runtime_key(self, entry: ModelEntry, *, device_preference: DevicePreference) -> str:
        """Build a cache key for the physical runtime artifact to reuse across roles."""
        runtime_mode = self._resolve_runtime_mode(entry, device_preference=device_preference)
        parts = [
            entry.backend,
            entry.repo_id,
            entry.revision or "",
            entry.filename or "",
            entry.tokenizer_repo or "",
            runtime_mode,
        ]
        return "||".join(parts)

    def _resolve_runtime_mode(
        self,
        entry: ModelEntry,
        *,
        device_preference: DevicePreference,
    ) -> str:
        """Collapse device preference to the actual runtime mode used for caching."""
        if device_preference == "cpu":
            return "cpu"
        diagnostics = get_runtime_diagnostics()
        if entry.backend == "transformers_safetensors":
            return "gpu" if diagnostics.transformers_gpu_ready() else "cpu"
        return "gpu" if diagnostics.gguf_gpu_ready() else "cpu"

    def _prepare_cpu_offload_switch(self, *, keep_runtime_key: str) -> None:
        """Offload inactive transformer backends to CPU RAM before switching."""
        for runtime_key, backend in list(self.loaded_models.items()):
            if runtime_key == keep_runtime_key:
                continue
            if hasattr(backend, "supports_cpu_offload") and backend.supports_cpu_offload():
                backend.offload_to_cpu()
                if self.logger is not None:
                    self.logger.emit(
                        "model_offloaded_to_cpu",
                        runtime_key=runtime_key,
                        runtime_summary=getattr(backend, "runtime_summary", "unknown"),
                        state="idle",
                    )
                continue
            self.loaded_models.pop(runtime_key, None)
            if self.logger is not None:
                self.logger.emit(
                    "model_unloaded_for_switch",
                    runtime_key=runtime_key,
                    state="idle",
                )
        unload_backend_resources()

    def _activate_backend(self, runtime_key: str, backend: TextGenerationBackend) -> None:
        """Move the selected backend into active inference residency when supported."""
        if not hasattr(backend, "supports_cpu_offload") or not backend.supports_cpu_offload():
            return
        backend.activate_for_inference()
        if self.logger is not None:
            self.logger.emit(
                "model_activated_for_inference",
                runtime_key=runtime_key,
                runtime_summary=getattr(backend, "runtime_summary", "unknown"),
                state="idle",
            )

    def _can_keep_cpu_offloaded(
        self,
        entries: list[ModelEntry],
        *,
        peak_active_vram_gb: float,
        total_estimated_vram_gb: float,
        snapshot: SystemMemorySnapshot,
        device_preference: DevicePreference,
    ) -> bool:
        """Return whether the selection can stay loaded with inactive models in CPU RAM."""
        if device_preference == "cpu" or not entries:
            return False
        diagnostics = get_runtime_diagnostics()
        if not diagnostics.transformers_gpu_ready():
            return False
        if any(entry.backend != "transformers_safetensors" for entry in entries):
            return False
        if snapshot.vram_available_gb is None or snapshot.ram_available_gb <= 0:
            return False
        return (
            peak_active_vram_gb <= snapshot.vram_available_gb * 0.85
            and total_estimated_vram_gb <= snapshot.ram_available_gb * 0.85
        )
