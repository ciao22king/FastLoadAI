"""Command-line interface for preparing and running local GGUF models."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .catalog import catalog_budget, catalog_match, catalog_rows
from .chat import run_chat
from .hardware import (
    detect_hardware,
    model_fit,
    recommend_settings,
    usable_model_memory_gb,
)
from .tui import (
    GREEN,
    RED,
    YELLOW,
    Theme,
    color_enabled,
    format_size,
    render_hardware,
    render_kv,
    render_plan,
    render_suggestions,
)

DEFAULT_CONFIG = Path("fastloadai-config.json")
QUANT_RANK = {
    "Q2": 20,
    "IQ2": 22,
    "Q3": 30,
    "IQ3": 32,
    "Q4": 40,
    "IQ4": 42,
    "Q5": 50,
    "Q6": 60,
    "Q8": 80,
    "F16": 90,
    "BF16": 90,
    "F32": 100,
}


def _write_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _quant_rank(filename: str) -> int:
    upper_name = filename.upper()
    for token, rank in QUANT_RANK.items():
        if re.search(rf"(?<![A-Z0-9]){token}(?:_[A-Z0-9]+)*", upper_name):
            return rank
    return 0


def _quant_label(filename: str) -> str:
    """Extract a short label such as ``Q4_K_M`` from a GGUF filename."""
    upper_name = filename.upper()
    for token in QUANT_RANK:
        match = re.search(rf"(?<![A-Z0-9]){token}(?:_[A-Z0-9]+)*", upper_name)
        if match:
            return match.group(0)
    return Path(filename).stem


PREFER_TARGET = {"speed": 40, "balanced": 45, "quality": 80}


def _quant_finesse(filename: str) -> int:
    """Prefer K-quants over legacy quants with the same nominal size."""
    upper_name = filename.upper()
    if "_K_M" in upper_name:
        return 3
    if "_K_S" in upper_name:
        return 2
    if re.search(r"_K(?![A-Z0-9])", upper_name):
        return 1
    return 0


def _gguf_groups(siblings: list[Any]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    shard_pattern = re.compile(r"^(.*?)-(\d{5})-of-(\d{5})(\.gguf)$", re.IGNORECASE)
    for sibling in siblings:
        name = sibling.rfilename
        if not name.lower().endswith(".gguf"):
            continue
        match = shard_pattern.match(name)
        group_name = match.group(1) + match.group(4) if match else name
        group = groups.setdefault(
            group_name,
            {"name": group_name, "files": [], "size": 0, "size_known": True},
        )
        group["files"].append(name)
        size = getattr(sibling, "size", None)
        if size is None:
            group["size_known"] = False
        else:
            group["size"] += size
    return list(groups.values())


def _select_gguf(
    siblings: list[Any],
    budget_gb: float,
    requested_file: str | None,
    prefer: str = "balanced",
) -> dict[str, Any]:
    groups = _gguf_groups(siblings)
    if requested_file:
        selected = next(
            (group for group in groups if requested_file == group["name"] or requested_file in group["files"]),
            None,
        )
        if selected is None:
            raise ValueError(f"GGUF file not found in repository: {requested_file}")
        return selected
    if not groups:
        raise ValueError("The Hugging Face repository does not contain any GGUF files.")

    fitting = [
        group
        for group in groups
        if group["size_known"] and group["size"] <= budget_gb * (1024**3)
    ]
    if fitting:
        target = PREFER_TARGET.get(prefer, PREFER_TARGET["balanced"])
        return min(
            fitting,
            key=lambda group: (
                abs(_quant_rank(group["name"]) - target),
                -_quant_finesse(group["name"]),
                group["size"],
            ),
        )

    known_sizes = [group for group in groups if group["size_known"]]
    if known_sizes:
        return min(known_sizes, key=lambda group: (group["size"], -_quant_rank(group["name"])))

    known_quants = [group for group in groups if _quant_rank(group["name"]) > 0]
    return min(known_quants or groups, key=lambda group: _quant_rank(group["name"]))


def _plan_command(args: argparse.Namespace) -> int:
    specs = detect_hardware()
    model_path = Path(args.model) if args.model else None
    config = recommend_settings(specs, model_path)
    if getattr(args, "low_memory", False):
        config["low_memory"] = True
        config["context_size"] = min(int(config["context_size"]), 1024)
    config["model_path"] = str(model_path.resolve()) if model_path else None
    _write_config(args.output, config)
    print(render_plan(config, str(args.output), Theme(color_enabled())))
    return 0


def _hf_token(args: argparse.Namespace) -> str | None:
    return (
        getattr(args, "token", None)
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )


def _download_command(args: argparse.Namespace) -> int:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        print("Install Hugging Face support with: pip install fastloadai[huggingface]", file=sys.stderr)
        return 2

    theme = Theme(color_enabled())
    specs = detect_hardware()
    budget_gb = usable_model_memory_gb(specs)
    token = _hf_token(args)
    revision = getattr(args, "revision", None)

    try:
        model_info = HfApi(token=token).model_info(
            args.repo, files_metadata=True, revision=revision
        )
        selected = _select_gguf(
            model_info.siblings,
            budget_gb,
            args.file,
            prefer=getattr(args, "prefer", "balanced"),
        )
    except Exception as error:
        message = str(error)
        if isinstance(error, ValueError) and "gguf" in message.lower():
            print(f"Cannot use {args.repo}: {message}", file=sys.stderr)
            print(
                "Choose a repository that contains GGUF files, usually named with a -GGUF suffix.",
                file=sys.stderr,
            )
            return 1
        print(f"Could not inspect {args.repo}: {message}", file=sys.stderr)
        lowered = message.lower()
        if "gated" in lowered or "401" in lowered or "403" in lowered:
            print(
                "This repository is gated. Accept its license on huggingface.co, then pass --token or set HF_TOKEN.",
                file=sys.stderr,
            )
        return 1

    selected_size_gb = selected["size"] / (1024**3) if selected["size_known"] else 0.0
    fit = model_fit(specs, selected_size_gb)
    rows = [
        ("Repository", args.repo),
        ("File", selected["name"]),
        ("Size", format_size(selected_size_gb) if selected["size_known"] else "unknown"),
        ("Memory budget", format_size(budget_gb)),
        ("Model fit", fit),
    ]
    if revision:
        rows.insert(1, ("Revision", revision))
    if getattr(args, "dry_run", False):
        print(render_kv("Download plan (dry run)", rows, theme), flush=True)
        if fit == "too_large":
            print(
                theme.paint(
                    "This file is larger than the estimated memory. Try a smaller model, --prefer speed, or --force.",
                    YELLOW,
                )
            )
        return 0

    explicit_file = bool(args.file)
    if fit == "too_large" and not explicit_file and not getattr(args, "force", False):
        print(render_kv("Download plan", rows, theme), flush=True)
        print(
            theme.paint(
                "No quantization fits this machine's estimated memory. Nothing was downloaded.",
                RED,
            ),
            file=sys.stderr,
        )
        print(
            "Try: fastloadai suggest, --prefer speed, a smaller repository, or --force to try anyway.",
            file=sys.stderr,
        )
        return 1

    if selected["size_known"] and specs.disk_free_gb:
        needed_gb = selected_size_gb * 1.05
        if needed_gb > specs.disk_free_gb:
            print(
                f"Not enough disk space: need about {format_size(needed_gb)}, "
                f"{format_size(specs.disk_free_gb)} free.",
                file=sys.stderr,
            )
            return 1

    if fit == "too_large":
        print(
            theme.paint(
                "Warning: this model is larger than the estimated memory. "
                "It will use mmap and may be slow or fail; try a smaller quantization.",
                YELLOW,
            ),
            file=sys.stderr,
        )

    try:
        model_paths = [
            hf_hub_download(
                repo_id=args.repo,
                filename=filename,
                local_dir=args.directory,
                token=token,
                revision=revision,
            )
            for filename in sorted(selected["files"])
        ]
    except Exception as error:
        message = str(error)
        print(f"Download failed: {message}", file=sys.stderr)
        if "gated" in message.lower() or "401" in message or "403" in message:
            print(
                "Accept the model license on huggingface.co, then pass --token or set HF_TOKEN.",
                file=sys.stderr,
            )
        return 1

    model_path = Path(model_paths[0])
    model_size_bytes = selected["size"] if selected["size_known"] else None
    quant_label = _quant_label(selected["name"])
    config = recommend_settings(specs, model_path, model_size_bytes=model_size_bytes)
    config.update(
        {
            "model_path": str(model_path.resolve()),
            "model_repo": args.repo,
            "revision": revision,
            "quantization": quant_label,
            "quantization_file": selected["name"],
            "model_size_bytes": model_size_bytes,
            "llama_cpp_moe": "Depends on architecture support in the installed llama.cpp build.",
        }
    )
    _write_config(args.output, config)
    rows = [
        ("Model", str(model_path)),
        ("Quantization", quant_label),
        ("Size", format_size(selected_size_gb) if selected["size_known"] else "unknown"),
        ("Model fit", str(config["model_fit"])),
        ("Context", str(config["context_size"])),
        ("Threads", str(config["cpu_threads"])),
        ("Config file", str(args.output)),
    ]
    print(render_kv("Model ready", rows, theme))
    return 0


def _is_repo_reference(value: str) -> bool:
    if not value or Path(value).exists() or value.lower().endswith(".gguf"):
        return False
    return "/" in value


def _download_for_run(args: argparse.Namespace, repo: str) -> int:
    download_args = argparse.Namespace(
        repo=repo,
        file=None,
        directory=args.models_dir,
        output=args.config,
        token=getattr(args, "token", None),
        revision=getattr(args, "revision", None),
        dry_run=False,
        force=getattr(args, "force", False),
        prefer=getattr(args, "prefer", "balanced"),
    )
    return _download_command(download_args)


def _resolve_run_model(
    requested_model: str | None,
    config: dict[str, Any],
    models_dir: Path,
) -> Path:
    configured_value = config.get("model_path")
    configured_path = Path(configured_value).expanduser() if configured_value else None

    if not requested_model:
        if configured_path is None:
            raise ValueError("No model configured: specify a model or run download first.")
        return configured_path

    requested_path = Path(requested_model).expanduser()
    if requested_path.is_file():
        return requested_path.resolve()

    query = requested_model.casefold()
    repo = str(config.get("model_repo", "")).casefold()
    if (
        configured_path
        and configured_path.is_file()
        and (query in configured_path.name.casefold() or query in repo)
    ):
        return configured_path

    matches = []
    if models_dir.is_dir():
        matches = [
            path
            for path in models_dir.rglob("*.gguf")
            if query in path.name.casefold() or query in str(path.parent).casefold()
        ]

    shard_pattern = re.compile(r"-\d{5}-of-\d{5}(?=\.gguf$)", re.IGNORECASE)
    grouped: dict[Path, Path] = {}
    for path in matches:
        group_key = path.with_name(shard_pattern.sub("", path.name))
        current = grouped.get(group_key)
        if current is None or path.name.casefold() < current.name.casefold():
            grouped[group_key] = path

    if len(grouped) == 1:
        return next(iter(grouped.values())).resolve()
    if len(grouped) > 1:
        choices = "\n".join(f"  {path}" for path in sorted(grouped.values()))
        raise ValueError(f"Ambiguous model name. Choose one of these GGUF files: {choices}")
    raise ValueError(f"Model '{requested_model}' was not found in the config or in {models_dir}.")


def _run_command(args: argparse.Namespace) -> int:
    config: dict[str, Any] = {}
    try:
        if args.config.is_file():
            loaded_config = json.loads(args.config.read_text(encoding="utf-8"))
            if not isinstance(loaded_config, dict):
                raise ValueError("The config file must contain a JSON object.")
            config = loaded_config

        requested_model = args.model
        model_path: Path | None = None
        if requested_model:
            try:
                model_path = _resolve_run_model(
                    requested_model, config, args.models_dir
                )
            except ValueError:
                model_path = None

            if model_path is None or not model_path.is_file():
                repo = requested_model if _is_repo_reference(requested_model) else None
                if repo is None:
                    matched = catalog_match(requested_model)
                    repo = matched.repo if matched else None
                if repo is None:
                    raise ValueError(
                        f"Model '{requested_model}' was not found. Use a downloaded "
                        "name, a Hugging Face repo ID, or a catalog name such as "
                        "qwen2.5-0.5b."
                    )
                if not shutil.which("llama-server"):
                    print(
                        "llama-server not found. Install llama.cpp before running a model.",
                        file=sys.stderr,
                    )
                    return 2
                print(f"Preparing {requested_model} for this machine...")
                code = _download_for_run(args, repo)
                if code:
                    return code
                config = json.loads(args.config.read_text(encoding="utf-8"))
                model_path = _resolve_run_model(None, config, args.models_dir)
        else:
            configured_value = config.get("model_path")
            configured_path = (
                Path(configured_value).expanduser() if configured_value else None
            )
            if configured_path is None or not configured_path.is_file():
                repo = str(config.get("model_repo") or "")
                if not repo:
                    raise ValueError(
                        "No model configured. Pass a model name, for example "
                        "'fastloadai run qwen2.5-0.5b', or run download first."
                    )
                if not shutil.which("llama-server"):
                    print(
                        "llama-server not found. Install llama.cpp before running a model.",
                        file=sys.stderr,
                    )
                    return 2
                print(f"Model file is missing. Downloading {repo} again.")
                code = _download_for_run(args, repo)
                if code:
                    return code
                config = json.loads(args.config.read_text(encoding="utf-8"))
            model_path = _resolve_run_model(None, config, args.models_dir)

        if model_path is None or not model_path.is_file():
            raise ValueError(f"Model file not found: {model_path}")
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"Invalid configuration: {error}", file=sys.stderr)
        return 2

    specs = detect_hardware()
    model_size_bytes = config.get("model_size_bytes")
    configured_model = config.get("model_path")
    if configured_model and Path(configured_model).expanduser().resolve() != model_path.resolve():
        model_size_bytes = None
    if not isinstance(model_size_bytes, int):
        model_size_bytes = None
    config.update(
        recommend_settings(specs, model_path, model_size_bytes=model_size_bytes)
    )
    if str(config.get("quantization", "")).lower().endswith(".gguf"):
        config["quantization"] = _quant_label(str(config["quantization"]))
    if getattr(args, "low_memory", False):
        config["low_memory"] = True
        if getattr(args, "ctx_size", None) is None:
            config["context_size"] = min(int(config["context_size"]), 1024)
    overrides = {
        "cpu_threads": getattr(args, "threads", None),
        "context_size": getattr(args, "ctx_size", None),
        "n_gpu_layers": getattr(args, "gpu_layers", None),
    }
    config.update({key: value for key, value in overrides.items() if value is not None})
    for key, minimum in (("cpu_threads", 1), ("context_size", 256), ("n_gpu_layers", 0)):
        if isinstance(config.get(key), int):
            config[key] = max(minimum, int(config[key]))
    config["model_path"] = str(model_path.resolve())
    _write_config(args.config, config)

    fit = str(config.get("model_fit", "unknown"))
    if fit == "too_large":
        print(
            Theme(color_enabled()).paint(
                "Warning: this model is larger than the estimated memory. "
                "Expect slow loading and possible failures; try a smaller quantization.",
                YELLOW,
            ),
            file=sys.stderr,
        )
    elif fit == "tight":
        print(
            Theme(color_enabled()).paint(
                "Note: this model is close to the memory limit. Close other apps if loading fails.",
                YELLOW,
            ),
            file=sys.stderr,
        )

    return run_chat(
        model_path,
        config,
        getattr(args, "prompt", None),
        color=False if getattr(args, "no_color", False) else None,
        temperature=getattr(args, "temperature", 0.7),
        max_tokens=getattr(args, "max_tokens", None),
        system_prompt=getattr(args, "system", None),
        once=getattr(args, "once", False),
    )


def _models_command(args: argparse.Namespace) -> int:
    root = args.models_dir
    if not root.is_dir():
        print(f"No models directory at {root}. Run download first.", file=sys.stderr)
        return 1

    groups: dict[str, dict[str, Any]] = {}
    shard_pattern = re.compile(r"-\d{5}-of-\d{5}(?=\.gguf$)", re.IGNORECASE)
    for path in sorted(root.rglob("*.gguf")):
        group_name = shard_pattern.sub("", path.name)
        entry = groups.setdefault(group_name, {"path": path, "size": 0})
        try:
            entry["size"] += path.stat().st_size
        except OSError:
            pass

    if not groups:
        print(f"No GGUF files found in {root}.")
        return 0

    theme = Theme(color_enabled())
    specs = detect_hardware()
    if getattr(args, "json", False):
        payload = [
            {
                "name": name,
                "path": str(entry["path"]),
                "size_bytes": entry["size"],
                "quantization": _quant_label(name),
                "fit": model_fit(specs, entry["size"] / (1024**3)),
            }
            for name, entry in sorted(groups.items())
        ]
        print(json.dumps(payload, indent=2))
        return 0

    rows = []
    for name, entry in sorted(groups.items()):
        fit = model_fit(specs, entry["size"] / (1024**3))
        style = GREEN if fit == "comfortable" else YELLOW if fit == "tight" else RED
        rows.append(
            (
                str(entry["path"].relative_to(root)),
                f"{_quant_label(name)} · {format_size(entry['size'] / (1024**3))} · {theme.paint(fit, style)}",
            )
        )
    print(render_kv("Local models", rows, theme))
    return 0


def _doctor_command(args: argparse.Namespace) -> int:
    theme = Theme(color_enabled())
    rows: list[tuple[str, str]] = []
    healthy = True

    version = sys.version.split()[0]
    python_ok = sys.version_info >= (3, 10)
    rows.append(("Python", f"{version} {theme.paint('ok', GREEN) if python_ok else theme.paint('need 3.10+', RED)}"))
    healthy = healthy and python_ok

    specs = detect_hardware()
    memory = format_size(specs.ram_gb)
    if specs.available_ram_gb:
        memory += f" ({format_size(specs.available_ram_gb)} free)"
    rows.append(("System RAM", memory))
    if specs.disk_free_gb:
        rows.append(("Disk free", format_size(specs.disk_free_gb)))
    if specs.gpu_name:
        kind = "unified memory" if specs.unified_memory else "VRAM"
        rows.append(("GPU", f"{specs.gpu_name} ({format_size(specs.gpu_vram_gb)} {kind})"))
    else:
        rows.append(("GPU", "none - CPU only"))
    rows.append(("Model budget", format_size(usable_model_memory_gb(specs))))

    server = shutil.which("llama-server")
    rows.append(
        (
            "llama-server",
            server if server else theme.paint("missing - install llama.cpp", RED),
        )
    )
    healthy = healthy and bool(server)

    quantizer = shutil.which("llama-quantize")
    rows.append(
        (
            "llama-quantize",
            quantizer if quantizer else theme.paint("optional, missing", YELLOW),
        )
    )

    try:
        import huggingface_hub

        hub_status = f"{huggingface_hub.__version__} {theme.paint('ok', GREEN)}"
    except ImportError:
        hub_status = theme.paint("optional, install fastloadai[huggingface]", YELLOW)
    rows.append(("Hugging Face", hub_status))

    config: dict[str, Any] = {}
    if args.config.is_file():
        try:
            loaded = json.loads(args.config.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                config = loaded
            rows.append(("Config", f"{args.config} {theme.paint('ok', GREEN)}"))
        except (OSError, json.JSONDecodeError) as error:
            rows.append(("Config", f"{args.config} {theme.paint(str(error), RED)}"))
            healthy = False
    else:
        rows.append(("Config", theme.paint("not found - run download or plan", YELLOW)))

    model_value = config.get("model_path")
    if model_value:
        model_path = Path(str(model_value)).expanduser()
        if model_path.is_file():
            rows.append(("Model", f"{model_path} {theme.paint('ok', GREEN)}"))
            size_gb = model_path.stat().st_size / (1024**3)
            fit = model_fit(specs, size_gb)
            style = GREEN if fit == "comfortable" else YELLOW if fit == "tight" else RED
            rows.append(("Model fit", theme.paint(fit, style)))
        else:
            rows.append(("Model", f"{model_path} {theme.paint('missing', RED)}"))
            healthy = False

    print(render_kv("Environment check", rows, theme))
    return 0 if healthy else 1


def _suggest_command(args: argparse.Namespace) -> int:
    specs = detect_hardware()
    rows = catalog_rows(specs)
    if getattr(args, "json", False):
        print(json.dumps(rows, indent=2))
        return 0

    theme = Theme(color_enabled())
    print(render_suggestions(rows, catalog_budget(specs), theme))
    recommended = next((row for row in rows if row["recommended"]), None)
    if recommended:
        print(
            "\nRun the recommended model with:\n"
            f"  fastloadai run {recommended['repo']}"
        )
    else:
        print(
            "\nNo catalog model fits comfortably. Try a smaller repository or "
            "download with --prefer speed --force."
        )
    return 0


def _quantize_command(args: argparse.Namespace) -> int:
    if not args.model.is_file():
        print(f"Model file not found: {args.model}", file=sys.stderr)
        return 2
    executable = shutil.which("llama-quantize")
    if not executable:
        print("llama-quantize not found. Install a llama.cpp build that includes the CLI tools.", file=sys.stderr)
        return 2
    try:
        return subprocess.run(
            [executable, str(args.model), str(args.output), args.type],
            check=False,
        ).returncode
    except OSError as error:
        print(f"Could not start llama-quantize: {error}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fastloadai", description="Prepare GGUF models for your local hardware.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    scan = commands.add_parser("scan", help="Show detected hardware specs.")
    scan.add_argument("--json", action="store_true", help="Print raw JSON instead of a table.")

    plan = commands.add_parser("plan", help="Build a runtime config for this machine.")
    plan.add_argument("model", nargs="?", help="Path to a local GGUF file.")
    plan.add_argument("--output", type=Path, default=DEFAULT_CONFIG)
    plan.add_argument(
        "--low-memory",
        action="store_true",
        help="Force a smaller context and small batches on constrained machines.",
    )

    download = commands.add_parser("download", help="Download a GGUF quantization from Hugging Face.")
    download.add_argument("repo", help="Repository ID, for example org/model-GGUF.")
    download.add_argument("--file", help="Explicit GGUF file or shard to download.")
    download.add_argument("--directory", type=Path, default=Path("models"))
    download.add_argument("--output", type=Path, default=DEFAULT_CONFIG)
    download.add_argument("--token", help="Hugging Face token; HF_TOKEN is used as a fallback.")
    download.add_argument("--revision", help="Hugging Face revision, branch, tag or commit.")
    download.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the selected file and memory fit without downloading.",
    )
    download.add_argument(
        "--force",
        action="store_true",
        help="Download even when the model does not fit the estimated memory.",
    )
    download.add_argument(
        "--prefer",
        choices=("speed", "balanced", "quality"),
        default="balanced",
        help="Quantization trade-off when several files fit. Default: balanced.",
    )

    suggest = commands.add_parser("suggest", help="Recommend small models that fit this machine.")
    suggest.add_argument("--json", action="store_true", help="Print raw JSON instead of a table.")

    models = commands.add_parser("models", help="List downloaded GGUF models.")
    models.add_argument("--models-dir", type=Path, default=Path("models"))
    models.add_argument("--json", action="store_true", help="Print raw JSON instead of a table.")

    doctor = commands.add_parser("doctor", help="Check the local runtime and configuration.")
    doctor.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    run = commands.add_parser("run", help="Apply the hardware plan and open an interactive chat.")
    run.add_argument(
        "model",
        nargs="?",
        help=(
            "Model name, GGUF path, or Hugging Face repo ID. "
            "Omit it to reuse the configured model."
        ),
    )
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--models-dir", type=Path, default=Path("models"))
    run.add_argument(
        "--prefer",
        choices=("speed", "balanced", "quality"),
        default="balanced",
        help="Quantization trade-off when run downloads a model. Default: balanced.",
    )
    run.add_argument("--token", help="Hugging Face token; HF_TOKEN is used as a fallback.")
    run.add_argument("--revision", help="Hugging Face revision for auto-downloads.")
    run.add_argument(
        "--force",
        action="store_true",
        help="Download even when the model does not fit the estimated memory.",
    )
    run.add_argument("--prompt", help="Send one message before the interactive prompt.")
    run.add_argument("--once", action="store_true", help="Answer the prompt and exit.")
    run.add_argument("--threads", type=int, help="Override CPU threads.")
    run.add_argument("--ctx-size", type=int, help="Override the context window.")
    run.add_argument("--gpu-layers", type=int, help="Override GPU layers (0 keeps CPU only).")
    run.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature. Default: 0.7.")
    run.add_argument("--max-tokens", type=int, help="Maximum tokens per answer.")
    run.add_argument("--system", help="Replace the default system prompt.")
    run.add_argument(
        "--low-memory",
        action="store_true",
        help="Force a smaller context and small batches on constrained machines.",
    )
    run.add_argument("--no-color", action="store_true", help="Disable colored output.")

    quantize = commands.add_parser("quantize", help="Quantize a local GGUF with llama-quantize.")
    quantize.add_argument("model", type=Path)
    quantize.add_argument("output", type=Path)
    quantize.add_argument("--type", default="Q4_K_M", help="llama.cpp type, for example Q4_K_M or Q5_K_M.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "scan":
        specs = detect_hardware()
        if getattr(args, "json", False):
            print(json.dumps(specs.__dict__, indent=2))
        else:
            print(render_hardware(specs, Theme(color_enabled())))
        return 0
    if args.command == "plan":
        return _plan_command(args)
    if args.command == "download":
        return _download_command(args)
    if args.command == "suggest":
        return _suggest_command(args)
    if args.command == "models":
        return _models_command(args)
    if args.command == "doctor":
        return _doctor_command(args)
    if args.command == "run":
        return _run_command(args)
    if args.command == "quantize":
        return _quantize_command(args)
    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())