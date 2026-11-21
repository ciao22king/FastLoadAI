# FastLoadAI

FastLoadAI is a Go command-line tool that downloads Hugging Face GGUF models,
fits them to the machine, and runs local inference through `llama.cpp`.

It detects the CPU, RAM, disk and GPU, chooses a quantization that fits, writes
a runtime plan, and opens a streaming chat through `llama-server`. Everything
runs locally: no prompt or model data leaves the machine.

## Features

- Hardware-aware model and quantization selection.
- Real file sizes from the Hugging Face API, including sharded GGUF files.
- One-command download and chat: `fastloadai run qwen2.5-0.5b`.
- Streaming chat with history trimming, retry, context usage and session stats.
- A polished terminal UI with a banner, spinner, colour themes and summaries.
- CPU and NVIDIA GPU detection, Apple Silicon unified memory support.
- Container-aware limits from cgroups, so Docker memory and CPU caps are honoured.
- `scan`, `suggest`, `download`, `plan`, `models`, `doctor`, `run` and `quantize`.
- JSON output for scripts on `scan`, `suggest` and `models`.

## Requirements

- Go 1.24 or newer to build from source.
- A `llama.cpp` build that provides `llama-server`.
- `llama-quantize` only for the optional `quantize` command.
- Git and curl for the one-line installer.

The installer downloads an official CPU build of `llama.cpp`. For GPU builds,
install `llama.cpp` yourself and keep `llama-server` on `PATH`.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/ciao22king/FastLoadAI/main/install.sh | sh
```

The installer clones the repository into `~/.local/share/fastloadai`, builds the
Go binary into `~/.local/bin/fastloadai`, and installs `llama-server`,
`llama-cli` and `llama-quantize` when they are missing.

Build manually:

```bash
git clone https://github.com/ciao22king/FastLoadAI.git
cd FastLoadAI
go build -o fastloadai ./cmd/fastloadai
./fastloadai --help
```

## Quick start

```bash
fastloadai scan
fastloadai suggest
fastloadai run qwen2.5-0.5b
```

`run` downloads the best fitting quantization when the model is not present,
then opens the chat. Ask a question, or use `/help` inside the chat.

```bash
fastloadai run qwen2.5-0.5b --prompt "Explain GGUF in one sentence." --once
```

## Commands

```bash
fastloadai scan
fastloadai scan --json
fastloadai suggest
fastloadai suggest --json
fastloadai download Qwen/Qwen2.5-0.5B-Instruct-GGUF
fastloadai download Qwen/Qwen2.5-0.5B-Instruct-GGUF --dry-run
fastloadai download Qwen/Qwen2.5-0.5B-Instruct-GGUF --prefer quality
fastloadai models
fastloadai models --json
fastloadai plan ./model.gguf --output config.json
fastloadai doctor
fastloadai run
fastloadai run qwen2.5-0.5b
fastloadai run Qwen/Qwen2.5-0.5B-Instruct-GGUF
fastloadai run --prompt "Introduce yourself." --once
fastloadai run --low-memory --ctx-size 1024
fastloadai run --ctx-size 8192 --threads 6 --temperature 0.4
fastloadai quantize ./model.gguf ./model-Q4_K_M.gguf --type Q4_K_M
fastloadai --version
```

### What each command does

`scan` prints the detected platform, CPU, CPU threads, total and free RAM, free
disk, GPU and the estimated model budget. `--json` returns the raw object.

`suggest` ranks catalog models by size and marks the best recommendation for
this machine. It prints the exact `run` command to copy.

`download` inspects the GGUF files in a Hugging Face repository, downloads the
best quantization that fits, and writes the runtime config. `--prefer balanced`
(default) targets a 4-5 bit K-quant, `speed` targets 4 bit, and `quality`
targets 8 bit. `--file` selects an exact file or shard. `--dry-run` shows the
choice and memory fit without downloading. Sharded models are downloaded
together and verified against the API size. If nothing fits, the command
refuses unless `--force` is set.

`models` lists downloaded GGUF files, grouped by shard, with size,
quantization and fit. `--json` is available for scripts.

`plan` writes a runtime config for this machine. Pass a local GGUF path to
include its size and fit.

`doctor` checks Go, RAM, disk, GPU, `llama-server`, `llama-quantize`, network
access to Hugging Face, the config file, the model path and the model fit. It
exits non-zero when a required piece is missing.

`run` resolves a catalog name, Hugging Face repo ID, recorded repo or local GGUF
path. If the model is missing, it downloads the best fit first, then starts
`llama-server` on `127.0.0.1` and opens the chat. Useful options:

- `--prefer speed|balanced|quality` controls automatic downloads.
- `--prompt` sends a first message; `--once` answers it and exits.
- `--threads`, `--ctx-size`, `--gpu-layers` override the plan.
- `--temperature`, `--max-tokens` and `--system` tune generation.
- `--low-memory` forces a small context and small batches.
- `--token`, `--revision` and `--force` work like `download`.
- `--no-color` disables ANSI colour for logs and CI.

`quantize` runs `llama-quantize` on an existing GGUF file.

## Chat commands

| Command | Action |
| --- | --- |
| `/help` | Show the command list |
| `/stats` | Show turns, tokens, speed and elapsed time |
| `/context` | Show context-window usage with a bar |
| `/retry` | Regenerate the last answer |
| `/clear` | Reset the conversation and session stats |
| `/exit`, `/quit` | Close the chat and stop the server |

`Ctrl+C` and `Ctrl+D` also end the chat. When a session ends, FastLoadAI prints
a compact summary with turns, tokens, speed and time.

History is trimmed by estimated token budget, not by a fixed message count, so
long conversations stay inside the context window. The chat has no file or
shell tools; it only sees the conversation.

## How the runtime plan works

Each config records the detected hardware and the recommended settings: CPU
threads, context size, mmap, model fit, low-memory mode and GPU offload.

The model budget uses currently available RAM, not total RAM, so it adapts to
other applications and to container limits. The context starts at 1K-8K
depending on free memory and doubles when the model leaves at least 4 GB of
headroom, capped at 8K. `mmap` lets `llama.cpp` map weights and load pages on
demand. The estimate is conservative, not a guarantee against running out of
memory.

GPU offload is enabled for NVIDIA GPUs when the model fits 60% of VRAM, and for
Apple Silicon unified memory when the model fits the budget. AMD and Intel GPUs
run on CPU, which is safe and still works.

## Docker

The image is CPU-only and supports `linux/amd64` and `linux/arm64`.

```bash
docker build -t fastloadai .
docker volume create fastloadai-data
docker run --rm -it -v fastloadai-data:/data fastloadai run qwen2.5-0.5b
```

The `/data` volume keeps models and `fastloadai-config.json` between runs. Add
`-e HF_TOKEN=...` for gated repositories. `--memory=4g` limits the container;
FastLoadAI reads the cgroup limit and chooses settings that fit. Use
`run --low-memory` on very small limits.

Pre-download without starting the chat:

```bash
docker run --rm -v fastloadai-data:/data fastloadai download Qwen/Qwen2.5-0.5B-Instruct-GGUF
```

The same workflow is available with Compose:

```bash
docker compose build
docker compose run --rm fastloadai run qwen2.5-0.5b
```

## Compatibility

FastLoadAI generates separate runtime configs and never modifies model files.
`quantize` wraps `llama-quantize`; converting Hugging Face checkpoints to GGUF
is not automated. MoE models work only when the installed `llama.cpp` build
supports the architecture.

GPU detection covers NVIDIA through `nvidia-smi` and Apple Silicon through
Metal. Other GPUs are treated as CPU-only.

## Verified with real models

The Go build was tested end to end on a 4-vCPU, 15.6 GB CPU-only container with
the official `llama.cpp` b12613 build:

| Model | Quant | Context | Speed |
| --- | --- | --- | --- |
| Qwen2.5-0.5B-Instruct | Q4_K_M | 8192 | ~20-25 tok/s |

Speeds are short-generation estimates and vary with prompt length, threads and
CPU.

## Development

```bash
go test ./...
go vet ./...
go build ./cmd/fastloadai
```

The module layout:

```
cmd/fastloadai/     CLI commands
internal/catalog/   built-in model catalog and ranking
internal/chat/      streaming chat loop and history trimming
internal/config/    config file read and write
internal/hardware/  CPU, RAM, disk, GPU and cgroup detection
internal/hf/        Hugging Face API, GGUF selection and downloads
internal/llama/     llama-server command, process and readiness
internal/model/     shared data types
internal/tui/       terminal UI, colour, tables, spinner and stats
```
