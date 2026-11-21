"""Curated GGUF models for machines with limited memory."""

from __future__ import annotations

from dataclasses import dataclass

from .hardware import HardwareSpecs, model_fit, usable_model_memory_gb


@dataclass(frozen=True)
class CatalogModel:
    name: str
    repo: str
    size_gb: float
    note: str


CATALOG: tuple[CatalogModel, ...] = (
    CatalogModel("SmolLM2 135M Instruct", "unsloth/SmolLM2-135M-Instruct-GGUF", 0.11, "tiny, fastest"),
    CatalogModel("SmolLM2 360M Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF", 0.25, "very small"),
    CatalogModel("Qwen2.5 0.5B Instruct", "Qwen/Qwen2.5-0.5B-Instruct-GGUF", 0.46, "small and capable"),
    CatalogModel("Llama 3.2 1B Instruct", "unsloth/Llama-3.2-1B-Instruct-GGUF", 0.80, "good balance"),
    CatalogModel("Qwen2.5 1.5B Instruct", "Qwen/Qwen2.5-1.5B-Instruct-GGUF", 1.00, "better quality"),
    CatalogModel("SmolLM2 1.7B Instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF", 1.10, "small general model"),
    CatalogModel("Llama 3.2 3B Instruct", "unsloth/Llama-3.2-3B-Instruct-GGUF", 2.00, "needs 6 GB+"),
    CatalogModel("Qwen2.5 3B Instruct", "Qwen/Qwen2.5-3B-Instruct-GGUF", 2.00, "needs 6 GB+"),
    CatalogModel("Qwen2.5 7B Instruct", "Qwen/Qwen2.5-7B-Instruct-GGUF", 4.70, "needs 10 GB+"),
    CatalogModel("Llama 3.1 8B Instruct", "unsloth/Llama-3.1-8B-Instruct-GGUF", 5.00, "needs 12 GB+"),
)


def catalog_rows(specs: HardwareSpecs) -> list[dict[str, object]]:
    """Return catalog models with a fit label and one recommendation."""
    rows = [
        {
            "name": model.name,
            "repo": model.repo,
            "size_gb": model.size_gb,
            "note": model.note,
            "fit": model_fit(specs, model.size_gb),
        }
        for model in CATALOG
    ]
    if specs.cpu_threads <= 2:
        cpu_cap = 0.5
    elif specs.cpu_threads <= 4:
        cpu_cap = 1.0
    elif specs.cpu_threads <= 8:
        cpu_cap = 2.0
    else:
        cpu_cap = 5.0
    comfortable = [
        row
        for row in rows
        if row["fit"] == "comfortable" and float(row["size_gb"]) <= cpu_cap
    ]
    recommended = max(comfortable, key=lambda row: float(row["size_gb"])) if comfortable else None
    for row in rows:
        row["recommended"] = row is recommended
    return sorted(rows, key=lambda row: float(row["size_gb"]))


def catalog_budget(specs: HardwareSpecs) -> float:
    return usable_model_memory_gb(specs)
