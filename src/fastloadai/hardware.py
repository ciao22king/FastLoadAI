"""Hardware discovery and conservative llama.cpp runtime recommendations."""

from __future__ import annotations

import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class HardwareSpecs:
    cpu_threads: int
    ram_gb: float
    gpu_name: str | None = None
    gpu_vram_gb: float = 0.0


def _total_ram_gb() -> float:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return pages * page_size / (1024**3)
    except (AttributeError, OSError, ValueError):
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024**2)
        except (OSError, ValueError, IndexError):
            pass
    return 0.0


def _nvidia_gpu() -> tuple[str | None, float]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=3,
        )
        first_gpu = result.stdout.splitlines()[0]
        name, memory_mb = first_gpu.rsplit(",", maxsplit=1)
        return name.strip(), float(memory_mb.strip()) / 1024
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return None, 0.0


def detect_hardware() -> HardwareSpecs:
    gpu_name, gpu_vram_gb = _nvidia_gpu()
    return HardwareSpecs(
        cpu_threads=max(1, os.cpu_count() or 1),
        ram_gb=round(_total_ram_gb(), 2),
        gpu_name=gpu_name,
        gpu_vram_gb=round(gpu_vram_gb, 2),
    )


def usable_model_memory_gb(specs: HardwareSpecs) -> float:
    """Estimate a conservative weights budget, preserving memory for the OS/KV cache."""
    ram_budget = max(0.0, specs.ram_gb - max(2.0, specs.ram_gb * 0.25)) * 0.75
    vram_budget = specs.gpu_vram_gb * 0.60
    return round(ram_budget + vram_budget, 2)


def recommended_context_size(specs: HardwareSpecs, model_size_gb: float = 0.0) -> int:
    """Pick the largest safe context, doubling it when the model leaves headroom."""
    if specs.ram_gb >= 24:
        context_size = 8192
    elif specs.ram_gb >= 12:
        context_size = 4096
    elif specs.ram_gb >= 8:
        context_size = 2048
    else:
        context_size = 1024

    if model_size_gb >= 0.1:
        headroom = usable_model_memory_gb(specs) - model_size_gb
        if headroom >= 4.0 and context_size < 8192:
            context_size *= 2
    return context_size


def recommend_settings(
    specs: HardwareSpecs,
    model_path: str | Path | None = None,
    model_size_bytes: int | None = None,
) -> dict[str, object]:
    model_size_gb = model_size_bytes / (1024**3) if model_size_bytes else 0.0
    if model_size_gb == 0.0 and model_path:
        path = Path(model_path)
        if path.is_file():
            model_size_gb = path.stat().st_size / (1024**3)

    context_size = recommended_context_size(specs, model_size_gb)

    gpu_layers = 0
    if (
        specs.gpu_name
        and model_size_gb > 0
        and model_size_gb <= specs.gpu_vram_gb * 0.60
    ):
        gpu_layers = 99

    return {
        "cpu_threads": max(1, min(specs.cpu_threads, 8)),
        "context_size": context_size,
        "n_gpu_layers": gpu_layers,
        "mmap": True,
        "model_size_gb": round(model_size_gb, 3),
        "usable_model_memory_gb": usable_model_memory_gb(specs),
        "hardware": asdict(specs),
    }