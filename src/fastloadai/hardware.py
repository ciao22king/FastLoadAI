"""Hardware discovery and conservative llama.cpp runtime recommendations."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class HardwareSpecs:
    cpu_threads: int
    ram_gb: float
    gpu_name: str | None = None
    gpu_vram_gb: float = 0.0
    available_ram_gb: float = 0.0
    disk_free_gb: float = 0.0
    unified_memory: bool = False
    is_laptop: bool = False


def _meminfo_gb(key: str) -> float:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{key}:"):
                return int(line.split()[1]) / (1024**2)
    except (OSError, ValueError, IndexError):
        pass
    return 0.0


def _total_ram_gb() -> float:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return pages * page_size / (1024**3)
    except (AttributeError, OSError, ValueError):
        return _meminfo_gb("MemTotal")


def _macos_available_ram_gb() -> float:
    try:
        total = int(
            subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                check=True,
                text=True,
                timeout=2,
            ).stdout.strip()
        ) / (1024**3)
        stats = subprocess.run(
            ["vm_stat"], capture_output=True, check=True, text=True, timeout=2
        ).stdout
        page_match = re.search(r"page size of (\d+) bytes", stats)
        page_size = int(page_match.group(1)) if page_match else 4096
        available_pages = 0
        for line in stats.splitlines():
            key, _, value = line.partition(":")
            if key.strip() in {"Pages free", "Pages inactive", "Pages speculative"}:
                pages = re.search(r"\d+", value)
                if pages:
                    available_pages += int(pages.group())
        if available_pages:
            return min(total, available_pages * page_size / (1024**3))
        return total
    except (OSError, subprocess.SubprocessError, ValueError):
        return _total_ram_gb()


def _available_ram_gb() -> float:
    available = _meminfo_gb("MemAvailable")
    if available:
        return available
    if platform.system() == "Darwin":
        return _macos_available_ram_gb()
    return _total_ram_gb()


def _disk_free_gb() -> float:
    try:
        return shutil.disk_usage(Path.cwd()).free / (1024**3)
    except OSError:
        return 0.0


def _is_laptop() -> bool:
    return any(Path("/sys/class/power_supply").glob("BAT*"))


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


def _apple_gpu(ram_gb: float) -> tuple[str | None, float, bool]:
    if platform.system() != "Darwin" or platform.machine() not in {"arm64", "aarch64"}:
        return None, 0.0, False
    # Apple Silicon uses unified memory: llama.cpp Metal offload shares system RAM.
    return "Apple Silicon (Metal)", round(ram_gb * 0.70, 2), True


def detect_hardware() -> HardwareSpecs:
    ram_gb = _total_ram_gb()
    gpu_name, gpu_vram_gb = _nvidia_gpu()
    unified_memory = False
    if not gpu_name:
        gpu_name, gpu_vram_gb, unified_memory = _apple_gpu(ram_gb)
    return HardwareSpecs(
        cpu_threads=max(1, os.cpu_count() or 1),
        ram_gb=round(ram_gb, 2),
        gpu_name=gpu_name,
        gpu_vram_gb=round(gpu_vram_gb, 2),
        available_ram_gb=round(_available_ram_gb(), 2),
        disk_free_gb=round(_disk_free_gb(), 2),
        unified_memory=unified_memory,
        is_laptop=_is_laptop(),
    )


def usable_model_memory_gb(specs: HardwareSpecs) -> float:
    """Estimate a conservative weights budget from currently available memory."""
    available = specs.available_ram_gb or specs.ram_gb
    floor = 1.5 if available >= 2.5 else 0.5
    reserve = min(max(floor, available * 0.25), available * 0.6)
    ram_budget = max(0.0, available - reserve) * 0.80
    vram_budget = 0.0 if specs.unified_memory else specs.gpu_vram_gb * 0.60
    return round(ram_budget + vram_budget, 2)


def model_fit(specs: HardwareSpecs, model_size_gb: float) -> str:
    """Classify a model as comfortable, tight, too large or unknown."""
    if model_size_gb < 0.001:
        return "unknown"
    budget = usable_model_memory_gb(specs)
    if budget <= 0:
        return "unknown"
    if model_size_gb <= budget * 0.75:
        return "comfortable"
    if model_size_gb <= budget:
        return "tight"
    return "too_large"


def recommended_context_size(specs: HardwareSpecs, model_size_gb: float = 0.0) -> int:
    """Pick the largest safe context, doubling it when the model leaves headroom."""
    available = specs.available_ram_gb or specs.ram_gb
    if available >= 24:
        context_size = 8192
    elif available >= 12:
        context_size = 4096
    elif available >= 8:
        context_size = 2048
    else:
        context_size = 1024

    budget = usable_model_memory_gb(specs)
    if model_size_gb > budget:
        return 512 if available < 6 else 1024

    if model_size_gb >= 0.01:
        headroom = budget - model_size_gb
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
    budget = usable_model_memory_gb(specs)
    fit = model_fit(specs, model_size_gb)

    gpu_layers = 0
    if specs.gpu_name and model_size_gb > 0:
        if specs.unified_memory:
            if model_size_gb <= budget:
                gpu_layers = 99
        elif model_size_gb <= specs.gpu_vram_gb * 0.60:
            gpu_layers = 99

    available = specs.available_ram_gb or specs.ram_gb
    low_memory = available < 6.0 or fit == "too_large"

    return {
        "cpu_threads": max(1, min(specs.cpu_threads, 8)),
        "context_size": context_size,
        "n_gpu_layers": gpu_layers,
        "mmap": True,
        "low_memory": low_memory,
        "model_fit": fit,
        "model_size_gb": round(model_size_gb, 3),
        "usable_model_memory_gb": budget,
        "hardware": asdict(specs),
    }
