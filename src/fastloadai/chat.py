"""Interactive local chat UI backed by llama.cpp's OpenAI-compatible server."""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MAX_HISTORY_MESSAGES = 21


def build_server_command(
    executable: str,
    model_path: Path,
    config: dict[str, Any],
    port: int,
) -> list[str]:
    command = [
        executable,
        "--model",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--threads",
        str(config["cpu_threads"]),
        "--ctx-size",
        str(config["context_size"]),
        "--n-gpu-layers",
        str(config["n_gpu_layers"]),
    ]
    if config.get("mmap") is False:
        command.append("--no-mmap")
    return command


def _available_port() -> int:
    with socket.socket() as server_socket:
        server_socket.bind(("127.0.0.1", 0))
        return int(server_socket.getsockname()[1])


def _wait_until_ready(process: subprocess.Popen[Any], base_url: str, timeout: float = 300) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("llama-server si è chiuso durante il caricamento del modello.")
        try:
            with urlopen(f"{base_url}/health", timeout=1) as response:
                if response.status == 200:
                    return
        except (HTTPError, URLError, TimeoutError, OSError):
            time.sleep(0.25)
    raise RuntimeError("Timeout: llama-server non è diventato disponibile.")


def stream_chat(base_url: str, messages: list[dict[str, str]]) -> Iterator[str]:
    payload = json.dumps(
        {
            "model": "local-model",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024,
            "stream": True,
        }
    ).encode("utf-8")
    request = Request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=300) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = event.get("choices", [])
                if choices:
                    content = choices[0].get("delta", {}).get("content")
                    if isinstance(content, str):
                        yield content
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"llama-server ha rifiutato la richiesta: {details}") from error
    except URLError as error:
        raise RuntimeError(f"Connessione al modello interrotta: {error.reason}") from error


def _print_help() -> None:
    print("Comandi: /help mostra questa guida, /clear azzera la conversazione, /exit chiude la chat.")


def _chat_loop(base_url: str, model_path: Path, initial_prompt: str | None) -> int:
    print()
    print(f"FASTLOADAI  |  {model_path.name}")
    print("Chat locale. Scrivi /help per i comandi, /exit per uscire.")
    messages = [
        {
            "role": "system",
            "content": "Sei un assistente utile. Rispondi in italiano salvo diversa richiesta. Hai accesso solo alla conversazione, non a file o shell.",
        }
    ]
    next_prompt = initial_prompt

    while True:
        if next_prompt is not None:
            prompt = next_prompt
            next_prompt = None
            print(f"\nTu > {prompt}")
        else:
            try:
                prompt = input("\nTu > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nChat terminata.")
                return 0

        if not prompt:
            continue
        if prompt in {"/exit", "/quit"}:
            print("Chat terminata.")
            return 0
        if prompt == "/help":
            _print_help()
            continue
        if prompt == "/clear":
            messages = messages[:1]
            print("Conversazione azzerata.")
            continue

        while len(messages) >= MAX_HISTORY_MESSAGES:
            del messages[1:3]
        messages.append({"role": "user", "content": prompt})
        print("\nFastLoadAI > ", end="", flush=True)
        response_parts = []
        try:
            for chunk in stream_chat(base_url, messages):
                print(chunk, end="", flush=True)
                response_parts.append(chunk)
        except RuntimeError as error:
            print(f"\nErrore: {error}")
            messages.pop()
            continue
        print()
        messages.append({"role": "assistant", "content": "".join(response_parts)})


def _stop_server(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _server_log_tail(log_file: Any) -> str:
    log_file.flush()
    log_file.seek(0, 2)
    log_file.seek(max(0, log_file.tell() - 3000))
    return log_file.read().strip()


def run_chat(model_path: Path, config: dict[str, Any], initial_prompt: str | None = None) -> int:
    executable = shutil.which("llama-server")
    if not executable:
        print("llama-server non trovato. Installa llama.cpp con il server incluso e rendilo disponibile nel PATH.", file=sys.stderr)
        return 2

    port = _available_port()
    base_url = f"http://127.0.0.1:{port}"
    command = build_server_command(executable, model_path, config, port)
    print(f"Caricamento del modello {model_path.name}...", flush=True)

    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as server_log:
        try:
            process = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT, text=True)
        except OSError as error:
            print(f"Impossibile avviare llama-server: {error}", file=sys.stderr)
            return 1

        try:
            _wait_until_ready(process, base_url)
            return _chat_loop(base_url, model_path, initial_prompt)
        except KeyboardInterrupt:
            print("\nChat terminata.")
            return 130
        except (OSError, RuntimeError) as error:
            print(f"Errore durante l'avvio della chat: {error}", file=sys.stderr)
            details = _server_log_tail(server_log)
            if details:
                print(details, file=sys.stderr)
            return 1
        finally:
            _stop_server(process)