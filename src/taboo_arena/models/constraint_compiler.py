"""Tokenizer-aware decode constraint compilation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from taboo_arena.models.registry import ModelEntry
from taboo_arena.utils.normalization import dedupe_preserve_order

_JSON_PREFIX_CANDIDATES = ["{", '{"', "{\n"]


@dataclass(slots=True, frozen=True)
class CompiledConstraints:
    """Backend-specific decode constraints compiled for one role/model call."""

    role: str
    model_id: str
    backend_name: str
    forbidden_surface_forms: list[str] = field(default_factory=list)
    json_output_required: bool = False
    supports_decode_enforcement: bool = False
    transformers_bad_words_ids: list[list[int]] = field(default_factory=list)
    gguf_banned_token_sequences: list[list[int]] = field(default_factory=list)
    json_prefix_token_sequences: list[list[int]] = field(default_factory=list)


class ConstraintCompiler:
    """Compile backend-specific decode constraints from surface-form inputs."""

    def compile(
        self,
        *,
        role: str,
        model_entry: ModelEntry,
        backend: Any,
        forbidden_surface_forms: list[str],
        json_output_required: bool,
    ) -> CompiledConstraints:
        """Compile advisory constraints for one selected role/model/backend."""
        normalized_surface_forms = _normalize_surface_forms(forbidden_surface_forms)
        if model_entry.backend == "transformers_safetensors":
            tokenizer = getattr(backend, "tokenizer", None)
            bad_words_ids = _compile_transformers_token_sequences(
                tokenizer,
                normalized_surface_forms,
            )
            json_prefix_token_sequences = (
                _compile_transformers_token_sequences(tokenizer, _JSON_PREFIX_CANDIDATES, variants=False)
                if json_output_required
                else []
            )
            return CompiledConstraints(
                role=role,
                model_id=model_entry.id,
                backend_name=model_entry.backend,
                forbidden_surface_forms=normalized_surface_forms,
                json_output_required=json_output_required,
                supports_decode_enforcement=bool(tokenizer),
                transformers_bad_words_ids=bad_words_ids,
                json_prefix_token_sequences=json_prefix_token_sequences,
            )

        llm = getattr(backend, "llm", None)
        gguf_banned_token_sequences = _compile_gguf_token_sequences(
            llm,
            normalized_surface_forms,
        )
        json_prefix_token_sequences = (
            _compile_gguf_token_sequences(llm, _JSON_PREFIX_CANDIDATES, variants=False)
            if json_output_required
            else []
        )
        return CompiledConstraints(
            role=role,
            model_id=model_entry.id,
            backend_name=model_entry.backend,
            forbidden_surface_forms=normalized_surface_forms,
            json_output_required=json_output_required,
            supports_decode_enforcement=False,
            gguf_banned_token_sequences=gguf_banned_token_sequences,
            json_prefix_token_sequences=json_prefix_token_sequences,
        )


def _normalize_surface_forms(surface_forms: list[str]) -> list[str]:
    return dedupe_preserve_order([item.strip() for item in surface_forms if item.strip()])


def _surface_form_variants(phrase: str) -> list[str]:
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


def _compile_transformers_token_sequences(
    tokenizer: Any,
    phrases: list[str],
    *,
    variants: bool = True,
) -> list[list[int]]:
    if tokenizer is None:
        return []
    blocked_sequences: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    for phrase in phrases:
        candidates = _surface_form_variants(phrase) if variants else [phrase]
        for candidate in candidates:
            try:
                token_ids = cast(list[int], tokenizer.encode(candidate, add_special_tokens=False))
            except Exception:
                continue
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


def _compile_gguf_token_sequences(
    llm: Any,
    phrases: list[str],
    *,
    variants: bool = True,
) -> list[list[int]]:
    tokenize = getattr(llm, "tokenize", None)
    if not callable(tokenize):
        return []
    token_sequences: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for phrase in phrases:
        candidates = _surface_form_variants(phrase) if variants else [phrase]
        for candidate in candidates:
            encoded = candidate.encode("utf-8")
            token_ids: list[int] | None = None
            try:
                token_ids = cast(list[int], tokenize(encoded, add_bos=False, special=False))
            except TypeError:
                try:
                    token_ids = cast(list[int], tokenize(encoded, add_bos=False))
                except Exception:
                    token_ids = None
            except Exception:
                token_ids = None
            if not token_ids:
                continue
            key = tuple(token_ids)
            if key in seen:
                continue
            seen.add(key)
            token_sequences.append(list(token_ids))
    return token_sequences
