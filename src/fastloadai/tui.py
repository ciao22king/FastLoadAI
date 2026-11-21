"""Terminal rendering helpers for the FastLoadAI chat interface.

Everything here is pure rendering: no network, no subprocesses, no input.
That keeps the visual layer easy to test and safe to reuse from the CLI.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, TextIO

from .hardware import HardwareSpecs, usable_model_memory_gb

RESET = "\033[0m"
BOLD = "1"
DIM = "2"
RED = "31"
GREEN = "32"
YELLOW = "33"
BLUE = "34"
MAGENTA = "35"
CYAN = "36"
GREY = "90"

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

HELP_ENTRIES: tuple[tuple[str, str], ...] = (
    ("/help", "Show this help"),
    ("/clear", "Reset the conversation history"),
    ("/stats", "Show session statistics"),
    ("/context", "Show context window usage"),
    ("/retry", "Regenerate the last answer"),
    ("/exit", "Close the chat and stop the server"),
)


def color_enabled(stream: TextIO | None = None) -> bool:
    """Return True when ANSI colour should be written to ``stream``."""
    if os.environ.get("NO_COLOR"):
        return False
    force = os.environ.get("FORCE_COLOR")
    if force is not None and force != "0":
        return True
    if os.environ.get("TERM", "") == "dumb":
        return False
    target = stream if stream is not None else sys.stdout
    return bool(getattr(target, "isatty", lambda: False)())


def visible_width(text: str) -> int:
    """Length of ``text`` without ANSI escape sequences."""
    return len(ANSI_RE.sub("", text))


def terminal_width(default: int = 80, minimum: int = 40, maximum: int = 96) -> int:
    """Terminal width clamped to a readable range."""
    try:
        columns = shutil.get_terminal_size((default, 24)).columns
    except OSError:
        columns = default
    return max(minimum, min(columns, maximum))


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 characters per token)."""
    return max(0, round(len(text) / 4))


def format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes)}m {remainder:02.0f}s"


def format_size(gigabytes: float) -> str:
    if gigabytes <= 0:
        return "unknown"
    if gigabytes < 1:
        return f"{gigabytes * 1024:.0f} MB"
    return f"{gigabytes:.1f} GB"


@dataclass(frozen=True)
class Theme:
    """ANSI palette that becomes a no-op when colour is disabled."""

    enabled: bool = True

    def paint(self, text: str, *styles: str) -> str:
        if not self.enabled or not styles or not text:
            return text
        return f"\033[{';'.join(styles)}m{text}{RESET}"


def _fit(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if visible_width(text) <= width:
        return text
    if width == 1:
        return "…"
    return text[: width - 1] + "…"


def render_kv(
    title: str,
    rows: Sequence[tuple[str, str]],
    theme: Theme,
    *,
    indent: str = "  ",
) -> str:
    """Render a small aligned key/value block."""
    if not rows:
        return theme.paint(title, BOLD)
    label_width = max(len(label) for label, _ in rows)
    lines = [theme.paint(title, BOLD)]
    for label, value in rows:
        lines.append(f"{indent}{theme.paint(label.ljust(label_width), CYAN)}  {value}")
    return "\n".join(lines)


def render_banner(
    model_name: str,
    config: Mapping[str, Any],
    *,
    theme: Theme | None = None,
    width: int | None = None,
) -> str:
    """Boxed header shown when a chat session starts."""
    theme = theme or Theme(True)
    width = max(32, width or terminal_width())
    inner = width - 4

    details = []
    if config.get("quantization"):
        details.append(str(config["quantization"]))
    if config.get("context_size"):
        details.append(f"ctx {config['context_size']}")
    if config.get("cpu_threads"):
        details.append(f"threads {config['cpu_threads']}")
    gpu_layers = config.get("n_gpu_layers")
    if isinstance(gpu_layers, int):
        details.append("CPU only" if gpu_layers == 0 else f"GPU layers {gpu_layers}")
    details.append("mmap on" if config.get("mmap", True) else "mmap off")
    model_size = config.get("model_size_gb")
    if isinstance(model_size, (int, float)) and model_size > 0:
        details.append(format_size(float(model_size)))

    def row(text: str, *styles: str) -> str:
        fitted = _fit(text, inner)
        padding = " " * max(0, inner - visible_width(fitted))
        return (
            theme.paint("│", GREY)
            + " "
            + theme.paint(fitted, *styles)
            + padding
            + " "
            + theme.paint("│", GREY)
        )

    top = theme.paint("╭" + "─" * inner + "╮", GREY)
    bottom = theme.paint("╰" + "─" * inner + "╯", GREY)
    lines = [
        top,
        row("FastLoadAI · local chat", BOLD, CYAN),
        row(model_name, BOLD),
        row(" · ".join(details), DIM),
        bottom,
    ]
    return "\n".join(lines)


def render_help(theme: Theme) -> str:
    command_width = max(len(command) for command, _ in HELP_ENTRIES)
    lines = [theme.paint("Commands", BOLD)]
    for command, description in HELP_ENTRIES:
        lines.append(
            f"  {theme.paint(command.ljust(command_width), CYAN)}  {description}"
        )
    lines.append("")
    lines.append(theme.paint("Ctrl+C or Ctrl+D also ends the chat.", DIM))
    return "\n".join(lines)


@dataclass
class ChatStats:
    """Accumulated usage for the current chat session."""

    turns: int = 0
    generated_chars: int = 0
    elapsed_seconds: float = 0.0

    def record(self, text: str, seconds: float) -> None:
        self.turns += 1
        self.generated_chars += len(text)
        self.elapsed_seconds += max(0.0, seconds)

    @property
    def estimated_tokens(self) -> int:
        return estimate_tokens("x" * self.generated_chars)

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.estimated_tokens / self.elapsed_seconds

    def render(self, theme: Theme) -> str:
        rows = [
            ("turns", str(self.turns)),
            ("generated", f"~{self.estimated_tokens} tokens"),
            ("speed", f"~{self.tokens_per_second:.1f} tok/s"),
            ("time", format_duration(self.elapsed_seconds)),
        ]
        return render_kv("Session", rows, theme)


def conversation_tokens(messages: Sequence[Mapping[str, Any]]) -> int:
    """Cheap estimate of the tokens currently held in the conversation."""
    return sum(estimate_tokens(str(message.get("content", ""))) for message in messages)


def render_context(used_tokens: int, limit: int, theme: Theme) -> str:
    """Show how much of the context window the conversation uses."""
    if limit <= 0:
        rows = [("used", f"~{used_tokens} tokens"), ("window", "unknown")]
        return render_kv("Context", rows, theme)
    ratio = min(1.0, max(0.0, used_tokens / limit))
    width = 24
    filled = round(ratio * width)
    bar = "█" * filled + "░" * (width - filled)
    style = GREEN if ratio < 0.7 else YELLOW if ratio < 0.9 else RED
    rows = [
        ("used", f"~{used_tokens} / {limit} tokens"),
        ("window", f"{theme.paint(bar, style)} {ratio:.0%}"),
    ]
    return render_kv("Context", rows, theme)


def render_timing(seconds: float, text: str, theme: Theme) -> str:
    tokens = estimate_tokens(text)
    rate = tokens / seconds if seconds > 0 else 0.0
    style = GREEN if rate >= 10 else YELLOW if rate >= 3 else RED
    detail = f"  {format_duration(seconds)} · ~{tokens} tokens · "
    return theme.paint(detail, DIM) + theme.paint(f"~{rate:.0f} tok/s", style)


def render_hardware(specs: HardwareSpecs, theme: Theme) -> str:
    rows = [
        ("CPU threads", str(specs.cpu_threads)),
        ("System RAM", format_size(specs.ram_gb)),
    ]
    if specs.gpu_name:
        rows.append(("GPU", f"{specs.gpu_name} ({format_size(specs.gpu_vram_gb)} VRAM)"))
    else:
        rows.append(("GPU", "none (CPU only)"))
    rows.append(("Model budget", format_size(usable_model_memory_gb(specs))))
    return render_kv("Hardware detected", rows, theme)


def render_plan(
    config: Mapping[str, Any],
    output: str,
    theme: Theme,
) -> str:
    model_path = config.get("model_path")
    rows = [
        ("Model", str(model_path) if model_path else "not specified"),
        ("Model size", format_size(float(config.get("model_size_gb") or 0))),
        ("Context", str(config.get("context_size", "unknown"))),
        ("Threads", str(config.get("cpu_threads", "unknown"))),
        ("GPU layers", str(config.get("n_gpu_layers", 0))),
        ("mmap", "on" if config.get("mmap", True) else "off"),
        ("Memory budget", format_size(float(config.get("usable_model_memory_gb") or 0))),
        ("Config file", output),
    ]
    return render_kv("Runtime plan", rows, theme)
