"""Interactive local chat UI backed by llama.cpp's OpenAI-compatible server."""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time

try:
    import readline  # noqa: F401 - enables arrow keys and history for input()
except ImportError:  # pragma: no cover - platform dependent
    pass

from pathlib import Path
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .tui import (
    BOLD,
    CYAN,
    DIM,
    GREEN,
    RED,
    YELLOW,
    ChatStats,
    Theme,
    color_enabled,
    conversation_tokens,
    estimate_tokens,
    render_banner,
    render_context,
    render_help,
    render_timing,
)

MAX_HISTORY_MESSAGES = 21

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer in English unless the user asks for another "
    "language. You can only see this conversation: you have no access to files or a shell."
)


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
    if config.get("low_memory"):
        command.extend(["--batch-size", "128", "--ubatch-size", "128"])
    return command


def _available_port() -> int:
    with socket.socket() as server_socket:
        server_socket.bind(("127.0.0.1", 0))
        return int(server_socket.getsockname()[1])


def _wait_until_ready(process: subprocess.Popen[Any], base_url: str, timeout: float = 300) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("llama-server exited while loading the model.")
        try:
            with urlopen(f"{base_url}/health", timeout=1) as response:
                if response.status == 200:
                    return
        except (HTTPError, URLError, TimeoutError, OSError):
            time.sleep(0.25)
    raise RuntimeError("Timed out waiting for llama-server to become ready.")


def stream_chat(
    base_url: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> Iterator[str]:
    payload = json.dumps(
        {
            "model": "local-model",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
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
                if data == "\x5bDONE\x5d":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = event.get("choices", [])
                if choices:
                    content = (choices[0].get("delta") or {}).get("content")
                    if isinstance(content, str):
                        yield content
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"llama-server rejected the request: {details}") from error
    except URLError as error:
        raise RuntimeError(f"Lost connection to the model: {error.reason}") from error


def trim_history(messages: list[dict[str, str]], budget_tokens: int) -> list[dict[str, str]]:
    """Drop the oldest user/assistant pairs until the estimate fits the budget."""
    while len(messages) > 2 and conversation_tokens(messages) > budget_tokens:
        del messages[1:3]
    return messages


def _chat_loop(
    base_url: str,
    initial_prompt: str | None,
    *,
    context_size: int = 0,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    system_prompt: str | None = None,
    once: bool = False,
    theme: Theme | None = None,
) -> int:
    theme = theme or Theme(color_enabled())
    interactive = bool(getattr(sys.stdout, "isatty", lambda: False)())
    one_shot = once or (initial_prompt is not None and not sys.stdin.isatty())
    token_limit = max_tokens or (min(1024, max(128, context_size // 4)) if context_size else 1024)
    history_budget = int(context_size * 0.6) if context_size else 0
    stats = ChatStats()
    messages = [{"role": "system", "content": system_prompt or SYSTEM_PROMPT}]
    next_prompt = initial_prompt
    user_label = theme.paint("you ›", BOLD, CYAN)
    assistant_label = theme.paint("fastloadai ›", BOLD, GREEN)
    while True:
        if next_prompt is not None:
            prompt = next_prompt
            next_prompt = None
            print(f"\n{user_label} {prompt}")
        else:
            try:
                prompt = input(f"\n{user_label} ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n" + theme.paint("Chat ended.", DIM))
                return 0
        if not prompt:
            continue
        command = prompt.casefold()
        if command in {"/exit", "/quit"}:
            print(theme.paint("Chat ended.", DIM))
            return 0
        if command == "/help":
            print()
            print(render_help(theme))
            continue
        if command == "/clear":
            messages = messages[:1]
            stats = ChatStats()
            print(theme.paint("Conversation cleared.", DIM))
            continue
        if command == "/stats":
            print()
            print(stats.render(theme))
            continue
        if command == "/context":
            print()
            print(render_context(conversation_tokens(messages), context_size, theme))
            continue
        if command == "/retry":
            if len(messages) < 2 or messages[-1]["role"] != "assistant":
                print(theme.paint("Nothing to retry yet.", DIM))
                continue
            last_user = messages[-2]["content"]
            messages.pop()
            messages.pop()
            next_prompt = last_user
            continue
        if prompt.startswith("/"):
            print(theme.paint(f"Unknown command {prompt}. Type /help for the list.", YELLOW))
            continue
        if history_budget:
            trim_history(messages, max(256, history_budget - estimate_tokens(prompt)))
        else:
            while len(messages) >= MAX_HISTORY_MESSAGES:
                del messages[1:3]
        messages.append({"role": "user", "content": prompt})
        print(f"\n{assistant_label} ", end="", flush=True)
        if interactive:
            print(theme.paint("thinking…", DIM), end="", flush=True)
        started = time.monotonic()
        parts: list[str] = []
        first_chunk = True
        interrupted = False
        try:
            for chunk in stream_chat(
                base_url,
                messages,
                temperature=temperature,
                max_tokens=token_limit,
            ):
                if first_chunk:
                    first_chunk = False
                    if interactive:
                        print("\r\033[K" + assistant_label + " ", end="", flush=True)
                print(chunk, end="", flush=True)
                parts.append(chunk)
        except KeyboardInterrupt:
            interrupted = True
            print()
        except RuntimeError as error:
            print(f"\n{theme.paint(str(error), RED)}")
            messages.pop()
            if one_shot:
                return 1
            continue
        elapsed = time.monotonic() - started
        text = "".join(parts)
        print()
        if not text:
            print(theme.paint("No response received.", DIM))
            messages.pop()
            if one_shot:
                return 1
            continue
        if interrupted:
            print(theme.paint("  (interrupted)", DIM))
        stats.record(text, elapsed)
        print(render_timing(elapsed, text, theme))
        messages.append({"role": "assistant", "content": text})
        if one_shot:
            return 0


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


def _is_memory_error(details: str) -> bool:
    lowered = details.lower()
    return any(
        marker in lowered
        for marker in (
            "failed to allocate",
            "out of memory",
            "not enough memory",
            "insufficient memory",
            "bad_alloc",
        )
    )


def run_chat(
    model_path: Path,
    config: dict[str, Any],
    initial_prompt: str | None = None,
    *,
    color: bool | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    system_prompt: str | None = None,
    once: bool = False,
) -> int:
    theme = Theme(color_enabled() if color is None else color)
    executable = shutil.which("llama-server")
    if not executable:
        print(
            theme.paint(
                "llama-server not found. Install a llama.cpp build that includes the server and put it on PATH.",
                RED,
            ),
            file=sys.stderr,
        )
        return 2

    print(render_banner(model_path.name, config, theme=theme))
    port = _available_port()
    base_url = f"http://127.0.0.1:{port}"
    command = build_server_command(executable, model_path, config, port)
    print(theme.paint("Loading model, this can take a while...", DIM), flush=True)

    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as server_log:
        try:
            process = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT, text=True)
        except OSError as error:
            print(f"Could not start llama-server: {error}", file=sys.stderr)
            return 1

        try:
            _wait_until_ready(process, base_url)
            print(theme.paint("Ready.", GREEN) + " " + theme.paint("Type /help for commands, /exit to quit.", DIM))
            return _chat_loop(
                base_url,
                initial_prompt,
                context_size=int(config.get("context_size") or 0),
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                once=once,
                theme=theme,
            )
        except KeyboardInterrupt:
            print("\n" + theme.paint("Chat ended.", DIM))
            return 130
        except (OSError, RuntimeError) as error:
            print(f"Error while starting the chat: {error}", file=sys.stderr)
            details = _server_log_tail(server_log)
            if details:
                print(details, file=sys.stderr)
            if details and _is_memory_error(details):
                print(
                    theme.paint(
                        "llama-server ran out of memory. Try --ctx-size 1024, "
                        "--prefer speed, or a smaller model (fastloadai suggest).",
                        YELLOW,
                    ),
                    file=sys.stderr,
                )
            return 1
        finally:
            _stop_server(process)
