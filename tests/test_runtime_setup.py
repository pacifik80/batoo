from __future__ import annotations

from taboo_arena.utils.runtime_setup import (
    RuntimeBootstrapReport,
    RuntimeProbe,
    plan_runtime_repairs,
)


def test_plan_runtime_repairs_prefers_gpu_when_system_gpu_is_present() -> None:
    plan = plan_runtime_repairs(
        True,
        RuntimeProbe(
            torch_version="2.11.0+cpu",
            torch_cuda_built=False,
            torch_cuda_available=False,
            llama_cpp_installed=True,
            llama_cpp_gpu_offload=False,
        ),
    )

    assert plan.prefer_gpu is True
    assert plan.desired_torch_backend == "auto"
    assert plan.repair_torch is True
    assert plan.repair_llama_cpp is False


def test_plan_runtime_repairs_uses_cpu_when_no_gpu_is_present() -> None:
    plan = plan_runtime_repairs(False, RuntimeProbe())

    assert plan.prefer_gpu is False
    assert plan.desired_torch_backend == "cpu"
    assert plan.repair_torch is True
    assert plan.repair_llama_cpp is True


def test_plan_runtime_repairs_skips_repairs_when_gpu_runtime_is_already_ready() -> None:
    plan = plan_runtime_repairs(
        True,
        RuntimeProbe(
            torch_version="2.11.0+cu130",
            torch_cuda_built=True,
            torch_cuda_available=True,
            torch_cuda_version="13.0",
            llama_cpp_installed=True,
            llama_cpp_gpu_offload=True,
        ),
    )

    assert plan.repair_torch is False
    assert plan.repair_llama_cpp is False


def test_runtime_bootstrap_report_serializes_probe_details() -> None:
    report = RuntimeBootstrapReport(
        system_gpu_present=True,
        system_gpu_name="RTX 4070 Ti",
        torch_backend_requested="auto",
        actions=["torch:auto"],
        warnings=[],
        final_probe=RuntimeProbe(
            torch_version="2.11.0+cu130",
            torch_cuda_built=True,
            torch_cuda_available=True,
            llama_cpp_installed=True,
            llama_cpp_gpu_offload=True,
        ),
    )

    payload = report.as_dict()

    assert payload["system_gpu_present"] is True
    assert payload["torch_backend_requested"] == "auto"
    assert payload["actions"] == ["torch:auto"]
    assert payload["final_probe"]["torch_version"] == "2.11.0+cu130"
