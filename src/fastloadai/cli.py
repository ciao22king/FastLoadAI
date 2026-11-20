"""Command-line interface for preparing and running local GGUF models."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .chat import run_chat
from .hardware import detect_hardware, recommend_settings, usable_model_memory_gb

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
        if re.search(rf"(?<![A-Z0-9]){token}(?:_[A-Z0-9]+)?", upper_name):
            return rank
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


def _select_gguf(siblings: list[Any], budget_gb: float, requested_file: str | None) -> dict[str, Any]:
    groups = _gguf_groups(siblings)
    if requested_file:
        selected = next(
            (group for group in groups if requested_file == group["name"] or requested_file in group["files"]),
            None,
        )
        if selected is None:
            raise ValueError(f"File GGUF non trovato nel repository: {requested_file}")
        return selected
    if not groups:
        raise ValueError("Il repository Hugging Face non contiene file GGUF.")

    fitting = [
        group
        for group in groups
        if group["size_known"] and group["size"] <= budget_gb * (1024**3)
    ]
    if fitting:
        return max(fitting, key=lambda group: _quant_rank(group["name"]))

    known_sizes = [group for group in groups if group["size_known"]]
    if known_sizes:
        return min(known_sizes, key=lambda group: (group["size"], -_quant_rank(group["name"])))

    known_quants = [group for group in groups if _quant_rank(group["name"]) > 0]
    return min(known_quants or groups, key=lambda group: _quant_rank(group["name"]))


def _plan_command(args: argparse.Namespace) -> int:
    specs = detect_hardware()
    model_path = Path(args.model) if args.model else None
    config = recommend_settings(specs, model_path)
    config["model_path"] = str(model_path.resolve()) if model_path else None
    _write_config(args.output, config)
    print(json.dumps(config, indent=2))
    print(f"Configurazione salvata in {args.output}")
    return 0


def _download_command(args: argparse.Namespace) -> int:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        print("Installa il supporto Hugging Face con: pip install 'fastloadai[huggingface]'", file=sys.stderr)
        return 2

    specs = detect_hardware()
    try:
        model_info = HfApi().model_info(args.repo, files_metadata=True)
        selected = _select_gguf(
            model_info.siblings,
            usable_model_memory_gb(specs),
            args.file,
        )
        if selected["size_known"] and selected["size"] > usable_model_memory_gb(specs) * (1024**3):
            print("Avviso: nessuna quantizzazione rientra nella memoria stimata; scarico il file richiesto/piu adatto.", file=sys.stderr)

        model_paths = [
            hf_hub_download(
                repo_id=args.repo,
                filename=filename,
                local_dir=args.directory,
                token=args.token,
            )
            for filename in sorted(selected["files"])
        ]
    except Exception as error:
        print(f"Download non riuscito: {error}", file=sys.stderr)
        return 1

    model_path = Path(model_paths[0])
    model_size_bytes = selected["size"] if selected["size_known"] else None
    config = recommend_settings(specs, model_path, model_size_bytes=model_size_bytes)
    config.update(
        {
            "model_path": str(model_path.resolve()),
            "model_repo": args.repo,
            "quantization": selected["name"],
            "model_size_bytes": model_size_bytes,
            "llama_cpp_moe": "Dipende dal supporto dell'architettura nella build llama.cpp installata.",
        }
    )
    _write_config(args.output, config)
    print(f"Modello: {model_path}")
    print(f"Quantizzazione: {selected['name']}")
    print(f"Configurazione per questo PC: {args.output}")
    return 0


def _resolve_run_model(
    requested_model: str | None,
    config: dict[str, Any],
    models_dir: Path,
) -> Path:
    configured_value = config.get("model_path")
    configured_path = Path(configured_value).expanduser() if configured_value else None

    if not requested_model:
        if configured_path is None:
            raise ValueError("Nessun modello configurato: specifica un modello o esegui prima download.")
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
        raise ValueError(f"Nome modello ambiguo. Scegli un file GGUF preciso:\n{choices}")
    raise ValueError(f"Modello '{requested_model}' non trovato nel config o in {models_dir}.")


def _run_command(args: argparse.Namespace) -> int:
    config: dict[str, Any] = {}
    try:
        if args.config.is_file():
            loaded_config = json.loads(args.config.read_text(encoding="utf-8"))
            if not isinstance(loaded_config, dict):
                raise ValueError("Il file di configurazione deve contenere un oggetto JSON.")
            config = loaded_config
        model_path = _resolve_run_model(args.model, config, args.models_dir)
        if not model_path.is_file():
            raise ValueError(f"File modello non trovato: {model_path}")
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"Configurazione non valida: {error}", file=sys.stderr)
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
    config["model_path"] = str(model_path.resolve())
    _write_config(args.config, config)

    return run_chat(model_path, config, args.prompt)


def _quantize_command(args: argparse.Namespace) -> int:
    if not args.model.is_file():
        print(f"File modello non trovato: {args.model}", file=sys.stderr)
        return 2
    executable = shutil.which("llama-quantize")
    if not executable:
        print("llama-quantize non trovato. Installa una build di llama.cpp con gli strumenti inclusi.", file=sys.stderr)
        return 2
    try:
        return subprocess.run(
            [executable, str(args.model), str(args.output), args.type],
            check=False,
        ).returncode
    except OSError as error:
        print(f"Impossibile avviare llama-quantize: {error}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fastloadai", description="Prepara modelli GGUF per l'hardware locale.")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("scan", help="Mostra le specifiche hardware rilevate.")

    plan = commands.add_parser("plan", help="Crea una configurazione runtime per questo PC.")
    plan.add_argument("model", nargs="?", help="Percorso di un file GGUF locale.")
    plan.add_argument("--output", type=Path, default=DEFAULT_CONFIG)

    download = commands.add_parser("download", help="Scarica una quantizzazione GGUF da Hugging Face.")
    download.add_argument("repo", help="ID del repository, ad esempio organizzazione/modello-GGUF.")
    download.add_argument("--file", help="File GGUF o shard da selezionare esplicitamente.")
    download.add_argument("--directory", type=Path, default=Path("models"))
    download.add_argument("--output", type=Path, default=DEFAULT_CONFIG)
    download.add_argument("--token", help="Token Hugging Face per repository gated o privati.")

    run = commands.add_parser("run", help="Applica la configurazione hardware e apre una chat interattiva.")
    run.add_argument("model", nargs="?", help="Nome del modello scaricato o percorso di un file GGUF.")
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--models-dir", type=Path, default=Path("models"))
    run.add_argument("--prompt")

    quantize = commands.add_parser("quantize", help="Quantizza un GGUF locale con llama-quantize.")
    quantize.add_argument("model", type=Path)
    quantize.add_argument("output", type=Path)
    quantize.add_argument("--type", default="Q4_K_M", help="Tipo llama.cpp, ad esempio Q4_K_M o Q5_K_M.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "scan":
        print(json.dumps(detect_hardware().__dict__, indent=2))
        return 0
    if args.command == "plan":
        return _plan_command(args)
    if args.command == "download":
        return _download_command(args)
    if args.command == "run":
        return _run_command(args)
    if args.command == "quantize":
        return _quantize_command(args)
    parser.error("Comando sconosciuto")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())