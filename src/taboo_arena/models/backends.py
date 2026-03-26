"""Model backend adapters."""

from __future__ import annotations

import gc
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, TypeVar, cast

from taboo_arena.config import GenerationParams
from taboo_arena.models.constraint_compiler import CompiledConstraints

T = TypeVar("T")


class ModelRuntimeError(RuntimeError):
    """Raised when a model cannot be loaded or used."""


@dataclass(slots=True)
class GenerationResponse:
    """Structured output from a backend call."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    prompt_template_id: str


@dataclass(slots=True, frozen=True)
class BackendRuntimeCapabilities:
    """Explicit runtime capability flags for one backend family."""

    backend_name: str
    supports_banned_phrase_enforcement: bool
    supports_cpu_offload: bool


class TextGenerationBackend(Protocol):
    """Backend protocol shared by all model families."""

    def generate(
        self,
        *,
        prompt: str,
        generation_params: GenerationParams,
        stop_tokens: list[str],
        prompt_template_id: str,
        banned_phrases: list[str] | None = None,
        compiled_constraints: CompiledConstraints | None = None,
    ) -> GenerationResponse:
        """Generate text from a rendered prompt."""

    def supports_cpu_offload(self) -> bool:
        """Return whether the backend can keep weights in CPU RAM while inactive."""

    def runtime_capabilities(self) -> BackendRuntimeCapabilities:
        """Return explicit capability flags for the current backend runtime."""

    def activate_for_inference(self) -> None:
        """Move the backend into its active inference residency."""

    def offload_to_cpu(self) -> None:
        """Move inactive weights to CPU RAM when supported."""


def truncate_on_stop_tokens(text: str, stop_tokens: list[str]) -> str:
    """Trim generated text at the earliest stop token."""
    if not stop_tokens:
        return text.strip()
    cutoff = len(text)
    for token in stop_tokens:
        if not token:
            continue
        index = text.find(token)
        if index != -1:
            cutoff = min(cutoff, index)
    return text[:cutoff].strip()


def _format_model_runtime_device(device: object) -> str:
    """Render a model device or device map entry in a stable way."""
    if isinstance(device, int):
        return f"cuda:{device}"
    return str(device)


def _supports_remote_code_fallback(source: str) -> bool:
    """Return whether retrying with trust_remote_code makes sense for this source."""
    path = Path(source)
    if not path.exists():
        return True
    candidate_root = path if path.is_dir() else path.parent
    return any(candidate_root.glob("*.py"))


def _load_pretrained_with_fallback(
    *,
    loader: Callable[..., T],
    source: str,
    kwargs: dict[str, object],
) -> T:
    """Try native Transformers loading first, then opt into remote code only when viable."""
    try:
        return loader(source, trust_remote_code=False, **kwargs)
    except Exception:
        if not _supports_remote_code_fallback(source):
            raise
    return loader(source, trust_remote_code=True, **kwargs)


def _bad_phrase_variants(phrase: str) -> list[str]:
    """Return phrase variants worth blocking exactly at generation time."""
    normalized = phrase.strip()
    if not normalized:
        return []
    candidates = {
        normalized,
        normalized.lower(),
        normalized.upper(),
        normalized.title(),
        f" {normalized}",
        f" {normalized.lower()}",
        f" {normalized.upper()}",
        f" {normalized.title()}",
    }
    return [candidate for candidate in candidates if candidate.strip()]


def _build_bad_words_ids(tokenizer: Any, phrases: list[str]) -> list[list[int]]:
    """Convert blocked phrases into token-id sequences for transformers generation."""
    blocked_sequences: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    for phrase in phrases:
        for variant in _bad_phrase_variants(phrase):
            token_ids = cast(list[int], tokenizer.encode(variant, add_special_tokens=False))
            if not token_ids:
                continue
            if eos_token_id is not None and any(token_id == eos_token_id for token_id in token_ids):
                continue
            key = tuple(token_ids)
            if key in seen:
                continue
            seen.add(key)
            blocked_sequences.append(list(token_ids))
    return blocked_sequences


class TransformersGenerator:
    """Transformers text generation backend."""

    def __init__(
        self,
        *,
        model_path: str,
        tokenizer_repo: str | None,
        device_preference: str,
        runtime_policy: str,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ModelRuntimeError(
                "Transformers backend requires 'transformers' and 'torch' to be installed."
            ) from exc

        tokenizer_source = tokenizer_repo or model_path
        self._torch = torch
        self._target_device = "cuda:0" if device_preference != "cpu" and torch.cuda.is_available() else "cpu"
        self._cpu_offload_enabled = (
            runtime_policy == "keep_cpu_offloaded_if_possible" and self._target_device != "cpu"
        )
        self._residency = "cpu"
        try:
            self.tokenizer = _load_pretrained_with_fallback(
                loader=AutoTokenizer.from_pretrained,
                source=tokenizer_source,
                kwargs={},
            )
            model_kwargs: dict[str, object] = {"torch_dtype": "auto"}
            if device_preference != "cpu" and not self._cpu_offload_enabled:
                model_kwargs["device_map"] = "auto"
            self.model = cast(
                Any,
                _load_pretrained_with_fallback(
                    loader=AutoModelForCausalLM.from_pretrained,
                    source=model_path,
                    kwargs=model_kwargs,
                ),
            )
            if self._cpu_offload_enabled:
                self.activate_for_inference()
            elif device_preference == "cpu":
                self.model.to("cpu")
                self._residency = "cpu"
            elif self._target_device != "cpu":
                self._residency = "gpu"
        except Exception as exc:
            raise ModelRuntimeError(f"Failed to load transformers model: {exc}") from exc
        self.runtime_summary = self._build_runtime_summary()

    def generate(
        self,
        *,
        prompt: str,
        generation_params: GenerationParams,
        stop_tokens: list[str],
        prompt_template_id: str,
        banned_phrases: list[str] | None = None,
        compiled_constraints: CompiledConstraints | None = None,
    ) -> GenerationResponse:
        try:
            started = perf_counter()
            inputs = self.tokenizer(prompt, return_tensors="pt")
            model_device = getattr(self.model, "device", None)
            if model_device is not None:
                inputs = {name: tensor.to(model_device) for name, tensor in inputs.items()}
            bad_words_ids = (
                list(compiled_constraints.transformers_bad_words_ids)
                if compiled_constraints is not None
                else _build_bad_words_ids(self.tokenizer, banned_phrases or [])
            )
            generation = self.model.generate(
                **inputs,
                do_sample=generation_params.temperature > 0,
                temperature=max(generation_params.temperature, 1e-5),
                top_p=generation_params.top_p,
                max_new_tokens=generation_params.max_tokens,
                pad_token_id=self.tokenizer.eos_token_id or self.tokenizer.pad_token_id,
                bad_words_ids=bad_words_ids or None,
            )
            prompt_token_count = int(inputs["input_ids"].shape[-1])
            completion_ids = generation[0][prompt_token_count:]
            completion_token_count = int(completion_ids.shape[-1])
            raw_text = cast(str, self.tokenizer.decode(completion_ids, skip_special_tokens=True))
            text = truncate_on_stop_tokens(raw_text, stop_tokens)
            latency_ms = round((perf_counter() - started) * 1000, 2)
            return GenerationResponse(
                text=text,
                prompt_tokens=prompt_token_count,
                completion_tokens=completion_token_count,
                latency_ms=latency_ms,
                prompt_template_id=prompt_template_id,
            )
        except Exception as exc:
            raise ModelRuntimeError(f"Transformers generation failed: {exc}") from exc

    def _build_runtime_summary(self) -> str:
        if self._cpu_offload_enabled:
            return f"transformers residency={self._residency} cpu_offload=on"
        hf_device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(hf_device_map, dict) and hf_device_map:
            values = sorted({_format_model_runtime_device(value) for value in hf_device_map.values()})
            return f"transformers device_map={','.join(values)}"
        model_device = getattr(self.model, "device", None)
        if model_device is not None:
            return f"transformers device={model_device}"
        return "transformers device=unknown"

    def supports_cpu_offload(self) -> bool:
        return self._cpu_offload_enabled

    def runtime_capabilities(self) -> BackendRuntimeCapabilities:
        return BackendRuntimeCapabilities(
            backend_name="transformers_safetensors",
            supports_banned_phrase_enforcement=True,
            supports_cpu_offload=self._cpu_offload_enabled,
        )

    def activate_for_inference(self) -> None:
        if not self._cpu_offload_enabled or self._residency == "gpu":
            return
        try:
            self.model.to(self._target_device)
            self._residency = "gpu"
            self.runtime_summary = self._build_runtime_summary()
        except Exception as exc:
            raise ModelRuntimeError(f"Failed to move transformers model to GPU: {exc}") from exc

    def offload_to_cpu(self) -> None:
        if not self._cpu_offload_enabled or self._residency == "cpu":
            return
        try:
            self.model.to("cpu")
            self._residency = "cpu"
            self.runtime_summary = self._build_runtime_summary()
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
        except Exception as exc:
            raise ModelRuntimeError(f"Failed to offload transformers model to CPU: {exc}") from exc


class LlamaCppGenerator:
    """llama.cpp GGUF backend."""

    def __init__(self, *, model_path: str, device_preference: str) -> None:
        try:
            import llama_cpp
            from llama_cpp import Llama
        except ImportError as exc:
            raise ModelRuntimeError(
                "GGUF backend requires 'llama-cpp-python' to be installed."
            ) from exc

        gpu_offload_supported = bool(llama_cpp.llama_cpp.llama_supports_gpu_offload())
        requested_gpu_layers = -1 if device_preference != "cpu" and gpu_offload_supported else 0
        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=4096,
                n_gpu_layers=requested_gpu_layers,
                verbose=False,
            )
        except Exception as exc:
            raise ModelRuntimeError(f"Failed to load GGUF model: {exc}") from exc
        self.runtime_summary = (
            "llama.cpp gpu_offload=on" if requested_gpu_layers != 0 else "llama.cpp gpu_offload=off"
        )

    def generate(
        self,
        *,
        prompt: str,
        generation_params: GenerationParams,
        stop_tokens: list[str],
        prompt_template_id: str,
        banned_phrases: list[str] | None = None,
        compiled_constraints: CompiledConstraints | None = None,
    ) -> GenerationResponse:
        try:
            started = perf_counter()
            response = cast(
                Any,
                self.llm.create_completion(
                    prompt=prompt,
                    max_tokens=generation_params.max_tokens,
                    temperature=generation_params.temperature,
                    top_p=generation_params.top_p,
                    stop=stop_tokens or None,
                ),
            )
            latency_ms = round((perf_counter() - started) * 1000, 2)
            choices = response.get("choices", [])
            text = ""
            if choices:
                text = truncate_on_stop_tokens(str(choices[0].get("text", "")), stop_tokens)
            usage = response.get("usage", {})
            return GenerationResponse(
                text=text,
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                latency_ms=latency_ms,
                prompt_template_id=prompt_template_id,
            )
        except Exception as exc:
            raise ModelRuntimeError(f"GGUF generation failed: {exc}") from exc

    def supports_cpu_offload(self) -> bool:
        return False

    def runtime_capabilities(self) -> BackendRuntimeCapabilities:
        return BackendRuntimeCapabilities(
            backend_name="llama_cpp_gguf",
            supports_banned_phrase_enforcement=False,
            supports_cpu_offload=False,
        )

    def activate_for_inference(self) -> None:
        return None

    def offload_to_cpu(self) -> None:
        return None


def unload_backend_resources() -> None:
    """Release Python and CUDA memory after unloading a model."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
