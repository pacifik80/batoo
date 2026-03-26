"""System inspection helpers."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from functools import lru_cache

import psutil


@dataclass(slots=True)
class SystemMemorySnapshot:
    """Simple system memory snapshot."""

    ram_total_gb: float
    ram_available_gb: float
    vram_total_gb: float | None
    vram_available_gb: float | None


@dataclass(slots=True)
class RuntimeDiagnostics:
    """GPU/runtime diagnostics for the local environment."""

    system_gpu_present: bool
    system_gpu_name: str | None
    nvidia_smi_visible: bool
    nvidia_driver_version: str | None
    torch_version: str | None
    torch_cuda_built: bool
    torch_cuda_available: bool
    torch_cuda_version: str | None
    llama_cpp_installed: bool
    llama_cpp_gpu_offload: bool

    def transformers_gpu_ready(self) -> bool:
        """Return whether the transformers backend can use CUDA."""
        return self.torch_cuda_built and self.torch_cuda_available

    def gguf_gpu_ready(self) -> bool:
        """Return whether the llama.cpp backend can offload to GPU."""
        return self.llama_cpp_installed and self.llama_cpp_gpu_offload

    def any_backend_gpu_ready(self) -> bool:
        """Return whether any supported backend can currently use GPU."""
        return self.transformers_gpu_ready() or self.gguf_gpu_ready()


@dataclass(slots=True)
class SystemLoadSample:
    """Point-in-time host and accelerator utilization percentages."""

    cpu_percent: float
    ram_percent: float
    gpu_percent: float | None
    vram_percent: float | None


def get_memory_snapshot() -> SystemMemorySnapshot:
    """Inspect available RAM and, when possible, CUDA VRAM."""
    vm = psutil.virtual_memory()
    vram_total: float | None = None
    vram_available: float | None = None
    try:
        import torch

        if torch.cuda.is_available():
            device_props = torch.cuda.get_device_properties(0)
            vram_total = round(device_props.total_memory / (1024**3), 2)
            allocated = torch.cuda.memory_allocated(0)
            vram_available = round((device_props.total_memory - allocated) / (1024**3), 2)
    except Exception:
        vram_total = None
        vram_available = None

    return SystemMemorySnapshot(
        ram_total_gb=round(vm.total / (1024**3), 2),
        ram_available_gb=round(vm.available / (1024**3), 2),
        vram_total_gb=vram_total,
        vram_available_gb=vram_available,
    )


def get_system_load_sample() -> SystemLoadSample:
    """Read current CPU/RAM/GPU utilization for compact dashboard widgets."""
    cpu_percent = round(float(psutil.cpu_percent(interval=None)), 1)
    ram_percent = round(float(psutil.virtual_memory().percent), 1)
    gpu_percent: float | None = None
    vram_percent: float | None = None

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        first_line = next(
            (line.strip() for line in completed.stdout.splitlines() if line.strip()),
            "",
        )
        if first_line:
            gpu_util_text, memory_used_text, memory_total_text = [
                item.strip() for item in first_line.split(",", maxsplit=2)
            ]
            gpu_percent = round(float(gpu_util_text), 1)
            memory_used = float(memory_used_text)
            memory_total = float(memory_total_text)
            if memory_total > 0:
                vram_percent = round((memory_used / memory_total) * 100.0, 1)
    except Exception:
        gpu_percent = None
        vram_percent = None

    return SystemLoadSample(
        cpu_percent=cpu_percent,
        ram_percent=ram_percent,
        gpu_percent=gpu_percent,
        vram_percent=vram_percent,
    )


@lru_cache(maxsize=1)
def get_runtime_diagnostics() -> RuntimeDiagnostics:
    """Inspect system GPU presence and backend runtime support."""
    gpu_name: str | None = None
    driver_version: str | None = None
    nvidia_smi_visible = False

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        first_line = next(
            (line.strip() for line in completed.stdout.splitlines() if line.strip()),
            "",
        )
        if first_line:
            nvidia_smi_visible = True
            parts = [part.strip() for part in first_line.split(",", maxsplit=1)]
            gpu_name = parts[0] if parts else None
            driver_version = parts[1] if len(parts) > 1 else None
    except Exception:
        nvidia_smi_visible = False

    torch_version: str | None = None
    torch_cuda_built = False
    torch_cuda_available = False
    torch_cuda_version: str | None = None
    try:
        import torch

        torch_version = str(torch.__version__)
        torch_cuda_built = bool(torch.backends.cuda.is_built())
        torch_cuda_available = bool(torch.cuda.is_available())
        torch_cuda_version = str(torch.version.cuda) if torch.version.cuda is not None else None
    except Exception:
        torch_version = None

    llama_cpp_installed = False
    llama_cpp_gpu_offload = False
    try:
        import llama_cpp

        llama_cpp_installed = True
        llama_cpp_gpu_offload = bool(llama_cpp.llama_cpp.llama_supports_gpu_offload())
    except Exception:
        llama_cpp_installed = False
        llama_cpp_gpu_offload = False

    system_gpu_present = nvidia_smi_visible or torch_cuda_available
    return RuntimeDiagnostics(
        system_gpu_present=system_gpu_present,
        system_gpu_name=gpu_name,
        nvidia_smi_visible=nvidia_smi_visible,
        nvidia_driver_version=driver_version,
        torch_version=torch_version,
        torch_cuda_built=torch_cuda_built,
        torch_cuda_available=torch_cuda_available,
        torch_cuda_version=torch_cuda_version,
        llama_cpp_installed=llama_cpp_installed,
        llama_cpp_gpu_offload=llama_cpp_gpu_offload,
    )
