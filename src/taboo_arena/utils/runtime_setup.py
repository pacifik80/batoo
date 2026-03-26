"""Runtime bootstrap helpers for CPU/GPU backend packages."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

TORCH_REQUIREMENT = "torch>=2.6.0"
LLAMA_CPP_REQUIREMENT = "llama-cpp-python>=0.3.5"
LLAMA_CPP_GPU_BOOTSTRAP_ENV = "TABOO_ARENA_BOOTSTRAP_LLAMA_CPP_GPU"
PROBE_SCRIPT = """
from __future__ import annotations

import json

payload = {
    "torch_version": None,
    "torch_cuda_built": False,
    "torch_cuda_available": False,
    "torch_cuda_version": None,
    "llama_cpp_installed": False,
    "llama_cpp_gpu_offload": False,
}
try:
    import torch

    payload["torch_version"] = str(torch.__version__)
    payload["torch_cuda_built"] = bool(torch.backends.cuda.is_built())
    payload["torch_cuda_available"] = bool(torch.cuda.is_available())
    payload["torch_cuda_version"] = str(torch.version.cuda) if torch.version.cuda else None
except Exception:
    pass
try:
    import llama_cpp

    payload["llama_cpp_installed"] = True
    payload["llama_cpp_gpu_offload"] = bool(llama_cpp.llama_cpp.llama_supports_gpu_offload())
except Exception:
    pass
print(json.dumps(payload))
""".strip()


@dataclass(slots=True)
class RuntimeProbe:
    """Observed backend package state inside the project environment."""

    torch_version: str | None = None
    torch_cuda_built: bool = False
    torch_cuda_available: bool = False
    torch_cuda_version: str | None = None
    llama_cpp_installed: bool = False
    llama_cpp_gpu_offload: bool = False

    def torch_gpu_ready(self) -> bool:
        return self.torch_cuda_built and self.torch_cuda_available

    def llama_gpu_ready(self) -> bool:
        return self.llama_cpp_installed and self.llama_cpp_gpu_offload


@dataclass(slots=True)
class RuntimePlan:
    """Desired runtime backend state for the current machine."""

    prefer_gpu: bool
    desired_torch_backend: Literal["auto", "cpu"]
    repair_torch: bool
    repair_llama_cpp: bool


@dataclass(slots=True)
class RuntimeBootstrapReport:
    """Summary of runtime bootstrap actions."""

    system_gpu_present: bool
    system_gpu_name: str | None
    torch_backend_requested: str
    actions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    final_probe: RuntimeProbe = field(default_factory=RuntimeProbe)

    def as_dict(self) -> dict[str, Any]:
        return {
            "system_gpu_present": self.system_gpu_present,
            "system_gpu_name": self.system_gpu_name,
            "torch_backend_requested": self.torch_backend_requested,
            "actions": list(self.actions),
            "warnings": list(self.warnings),
            "final_probe": asdict(self.final_probe),
        }


def detect_nvidia_gpu() -> tuple[bool, str | None]:
    """Return whether `nvidia-smi` reports a visible GPU."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return False, None

    gpu_name = next((line.strip() for line in completed.stdout.splitlines() if line.strip()), None)
    return gpu_name is not None, gpu_name


def plan_runtime_repairs(system_gpu_present: bool, probe: RuntimeProbe) -> RuntimePlan:
    """Decide whether the env should be repaired for CPU or GPU use."""
    prefer_gpu = system_gpu_present
    desired_torch_backend: Literal["auto", "cpu"] = "auto" if prefer_gpu else "cpu"
    repair_torch = probe.torch_version is None or (prefer_gpu and not probe.torch_gpu_ready())
    repair_llama_cpp = not probe.llama_cpp_installed
    return RuntimePlan(
        prefer_gpu=prefer_gpu,
        desired_torch_backend=desired_torch_backend,
        repair_torch=repair_torch,
        repair_llama_cpp=repair_llama_cpp,
    )


def probe_runtime(*, python_executable: Path, repo_root: Path) -> RuntimeProbe:
    """Inspect torch and llama.cpp support inside the selected interpreter."""
    completed = subprocess.run(
        [str(python_executable), "-c", PROBE_SCRIPT],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    payload = json.loads(completed.stdout)
    return RuntimeProbe(
        torch_version=payload.get("torch_version"),
        torch_cuda_built=bool(payload.get("torch_cuda_built", False)),
        torch_cuda_available=bool(payload.get("torch_cuda_available", False)),
        torch_cuda_version=payload.get("torch_cuda_version"),
        llama_cpp_installed=bool(payload.get("llama_cpp_installed", False)),
        llama_cpp_gpu_offload=bool(payload.get("llama_cpp_gpu_offload", False)),
    )


def ensure_runtime_environment(
    repo_root: Path,
    *,
    python_executable: Path | None = None,
    echo: Callable[[str], None] = print,
) -> RuntimeBootstrapReport:
    """Repair the project env so it uses GPU when available, else CPU."""
    interpreter = python_executable or Path(sys.executable)
    system_gpu_present, gpu_name = detect_nvidia_gpu()
    initial_probe = probe_runtime(python_executable=interpreter, repo_root=repo_root)
    plan = plan_runtime_repairs(system_gpu_present, initial_probe)
    report = RuntimeBootstrapReport(
        system_gpu_present=system_gpu_present,
        system_gpu_name=gpu_name,
        torch_backend_requested=plan.desired_torch_backend,
    )

    target = f"GPU ({gpu_name})" if system_gpu_present and gpu_name else "CPU"
    echo(f"[taboo-arena:bootstrap] Preparing runtime for {target}.")

    if plan.repair_torch:
        _install_torch(
            interpreter=interpreter,
            repo_root=repo_root,
            torch_backend=plan.desired_torch_backend,
            actions=report.actions,
            echo=echo,
        )

    if plan.repair_llama_cpp:
        _install_llama_cpp(
            interpreter=interpreter,
            repo_root=repo_root,
            prefer_gpu=plan.prefer_gpu,
            actions=report.actions,
            warnings=report.warnings,
            echo=echo,
        )

    final_probe = probe_runtime(python_executable=interpreter, repo_root=repo_root)
    report.final_probe = final_probe

    if plan.prefer_gpu and not final_probe.torch_gpu_ready():
        report.warnings.append(
            "Torch is still not CUDA-ready after bootstrap. Transformers models will stay on CPU."
        )
    if plan.prefer_gpu and not final_probe.llama_gpu_ready():
        report.warnings.append(
            "llama-cpp-python still lacks GPU offload after bootstrap. GGUF models will stay on CPU."
        )

    echo(
        "[taboo-arena:bootstrap] Final runtime: "
        f"torch={final_probe.torch_version or 'missing'} "
        f"(cuda_ready={final_probe.torch_gpu_ready()}, cuda={final_probe.torch_cuda_version or 'n/a'}); "
        f"llama_cpp_installed={final_probe.llama_cpp_installed}, "
        f"llama_gpu_ready={final_probe.llama_gpu_ready()}."
    )
    for warning in report.warnings:
        echo(f"[taboo-arena:bootstrap] Warning: {warning}")
    return report


def _install_torch(
    *,
    interpreter: Path,
    repo_root: Path,
    torch_backend: Literal["auto", "cpu"],
    actions: list[str],
    echo: Callable[[str], None],
) -> None:
    echo(f"[taboo-arena:bootstrap] Installing torch backend '{torch_backend}'.")
    _run_command(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(interpreter),
            "--upgrade",
            "--reinstall-package",
            "torch",
            "--torch-backend",
            torch_backend,
            TORCH_REQUIREMENT,
        ],
        cwd=repo_root,
        timeout=900,
    )
    actions.append(f"torch:{torch_backend}")


def _install_llama_cpp(
    *,
    interpreter: Path,
    repo_root: Path,
    prefer_gpu: bool,
    actions: list[str],
    warnings: list[str],
    echo: Callable[[str], None],
) -> None:
    if prefer_gpu and os.getenv(LLAMA_CPP_GPU_BOOTSTRAP_ENV) == "1":
        echo("[taboo-arena:bootstrap] Building llama-cpp-python with CUDA support.")
        env = os.environ.copy()
        env["CMAKE_ARGS"] = "-DGGML_CUDA=on"
        env["FORCE_CMAKE"] = "1"
        try:
            _run_command(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(interpreter),
                    "--upgrade",
                    "--reinstall-package",
                    "llama-cpp-python",
                    "--no-binary",
                    "llama-cpp-python",
                    LLAMA_CPP_REQUIREMENT,
                ],
                cwd=repo_root,
                env=env,
                timeout=1800,
            )
            probe = probe_runtime(python_executable=interpreter, repo_root=repo_root)
            if probe.llama_gpu_ready():
                actions.append("llama-cpp-python:source-cuda-build")
                return
        except subprocess.CalledProcessError:
            warnings.append("Automatic llama-cpp CUDA build failed; installing the default build instead.")

    echo("[taboo-arena:bootstrap] Installing llama-cpp-python.")
    _run_command(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(interpreter),
            "--upgrade",
            "--reinstall-package",
            "llama-cpp-python",
            LLAMA_CPP_REQUIREMENT,
        ],
        cwd=repo_root,
        timeout=900,
    )
    actions.append("llama-cpp-python:default")
    if prefer_gpu:
        warnings.append(
            "GGUF GPU bootstrap is disabled by default. Set "
            f"{LLAMA_CPP_GPU_BOOTSTRAP_ENV}=1 before launch to attempt a CUDA build."
        )


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout: int,
) -> None:
    subprocess.run(command, cwd=cwd, check=True, env=env, timeout=timeout)
