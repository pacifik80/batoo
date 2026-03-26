"""Configuration models."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from taboo_arena.utils.paths import get_log_dir

RoleName = Literal["cluer", "guesser", "judge"]
BackendName = Literal["transformers_safetensors", "llama_cpp_gguf"]
MemoryPolicy = Literal[
    "keep_loaded_if_possible",
    "keep_cpu_offloaded_if_possible",
    "sequential_load_unload",
]
DevicePreference = Literal["auto", "cpu", "cuda"]


class GenerationParams(BaseModel):
    """Generation parameters for a single role."""

    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    max_tokens: int = Field(default=128, ge=1, le=4096)


class RoleGenerationSettings(BaseModel):
    """Per-role generation settings."""

    cluer: GenerationParams = Field(
        default_factory=lambda: GenerationParams(temperature=0.4, max_tokens=256)
    )
    guesser: GenerationParams = Field(default_factory=lambda: GenerationParams(temperature=0.2))
    judge: GenerationParams = Field(
        default_factory=lambda: GenerationParams(temperature=0.1, max_tokens=384)
    )


class LogicalValidatorSettings(BaseModel):
    """Strictness toggles for the deterministic validator."""

    min_token_count: int = Field(default=2, ge=1, le=10)
    check_substring: bool = True
    check_token_match: bool = True
    check_whole_word: bool = True
    check_stemming: bool = True
    check_similarity_to_previous: bool = True
    similarity_threshold: int = Field(default=88, ge=1, le=100)


class DatasetSettings(BaseModel):
    """Dataset configuration."""

    source_repo: str = "Kovah/Taboo-Data"
    source_ref: str = "main"
    language: str = "en"


class RunSettings(BaseModel):
    """Runtime configuration for a single benchmark run."""

    max_guess_attempts: int = Field(default=3, ge=1, le=10)
    max_clue_repairs: int = Field(default=3, ge=1, le=10)
    guesser_hidden_retry_budget: int = Field(default=2, ge=0, le=10)
    block_on_uncertain: bool = False
    memory_policy: MemoryPolicy = "keep_loaded_if_possible"
    random_seed: int = 7
    device_preference: DevicePreference = "auto"
    show_hidden_repairs: bool = True
    allow_same_model_for_multiple_roles: bool = True
    debug_show_target: bool = False
    console_trace: bool = True
    generation: RoleGenerationSettings = Field(default_factory=RoleGenerationSettings)
    logical_validator: LogicalValidatorSettings = Field(default_factory=LogicalValidatorSettings)
    log_dir: Path = Field(default_factory=get_log_dir)


class AppSettings(BaseModel):
    """Top-level app settings."""

    dataset: DatasetSettings = Field(default_factory=DatasetSettings)
    run: RunSettings = Field(default_factory=RunSettings)

    @classmethod
    def from_yaml(cls, path: Path) -> AppSettings:
        """Load settings from a YAML file."""
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.model_validate(payload)
