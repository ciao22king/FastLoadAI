# FastLoadAI

FastLoadAI is a Go command-line tool that downloads Hugging Face GGUF models,
fits them to the machine, and runs local inference through `llama.cpp`.

It detects the CPU, RAM, disk and GPU, chooses a quantization that fits, writes
a runtime plan, and opens a streaming chat through `llama-server`. Everything
runs locally: no prompt or model data leaves the machine.

## Features

- Hardware-aware model and quantization selection.
- Real file sizes from the Hugging Face API, including sharded GGUF files.
- Parallel shard downloads with an auto-tuned worker pool (up to 16 workers).
- Hardware probing, Hugging Face metadata and doctor checks run concurrently.
- One-command download and chat: `fastloadai run qwen2.5-0.5b`.
- Automatic optimization profiles: `auto`, `speed`, `balanced`, `memory` and `quality`.
- MoE-aware CPU expert offload, GPU expert cache and tensor placement overrides.
- Flash Attention, split K/V cache quantization, unified KV, prompt-cache RAM limits and slot persistence.
- RoPE scaling/YaRN, SWA checkpoints, CPU affinity for hybrid P/E-core CPUs and startup tuning.
- Speculative decoding with ngram lookup or a separate draft model.
- Batch/ubatch, parallel slots, thread-batch, NUMA, GPU split and priority tuning.
- Full-screen chat UI with streaming, scrollback, input history and a status bar.
- Streaming chat with history trimming, retry, context usage and session stats.
- A polished terminal UI with a banner, spinner, colour themes and summaries.
- CPU and NVIDIA GPU detection, Apple Silicon unified memory support.
- Container-aware limits from cgroups, so Docker memory and CPU caps are honoured.
- `scan`, `suggest`, `download`, `plan`, `optimize`, `models`, `doctor`, `run` and `quantize`.
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

### Prebuilt packages

Every tagged release publishes Linux tarballs and Debian packages. The
`Packages` workflow also publishes an unsigned APT repository on GitHub Pages.

#### Termux

The package is submitted to TUR. Once it is accepted:

```bash
pkg install tur-repo
pkg install fastloadai
```

You can also download the `.deb` for your architecture from the latest GitHub
release and install it directly:

```bash
pkg install ./fastloadai_<version>_aarch64.deb
```

Use the `x86_64` file on Intel Chromebooks and Android emulators. Or use the
GitHub Pages APT repository:

```bash
echo "deb [trusted=yes] https://ciao22king.github.io/FastLoadAI/termux stable main" \
  > "$PREFIX/etc/apt/sources.list.d/fastloadai.list"
pkg update
pkg install fastloadai
```

The Termux package depends on `llama-cpp`, which provides `llama-server` and
`llama-quantize`.

#### Debian and Ubuntu

```bash
echo "deb [trusted=yes] https://ciao22king.github.io/FastLoadAI/debian stable main" \
  | sudo tee /etc/apt/sources.list.d/fastloadai.list
sudo apt update
sudo apt install fastloadai
```

The Debian package contains only the FastLoadAI binary. Install `llama.cpp`
separately so `llama-server` is on `PATH`.

The APT repositories are not GPG-signed, so `[trusted=yes]` is required. Use
them only if you trust this repository and the GitHub Pages deployment.

#### Go

```bash
go install github.com/ciao22king/FastLoadAI/cmd/fastloadai@latest
```

### Nix

The repository is a flake:

```bash
nix run github:ciao22king/FastLoadAI
nix build .#fastloadai
nix profile install .#fastloadai
```

Inside a checkout, `nix develop` provides Go and `gopls`. The package builds
with `CGO_ENABLED=0`, so the result is a static binary.

### Termux (Android)

For `pkg` packages, see [Prebuilt packages](#termux) above. To build the
Termux recipe yourself, copy it into a `termux-packages` checkout:

```bash
cp -a packaging/termux/fastloadai /path/to/termux-packages/packages/
cd /path/to/termux-packages
./build-package.sh fastloadai
```

From a non-Android host, use the Termux Docker wrapper:

```bash
./scripts/run-docker.sh ./build-package.sh fastloadai
```

The package depends on `llama-cpp`, which provides `llama-server` and
`llama-quantize`. It is limited to 64-bit Termux because `llama-cpp` does not
support 32-bit Android.

## Quick start

```bash
fastloadai scan
fastloadai suggest
fastloadai run qwen2.5-0.5b
```

`run` downloads the best fitting quantization when the model is not present,
then opens the full-screen chat. Ask a question, or use `/help` inside the chat.

```bash
fastloadai run qwen2.5-0.5b --prompt "Explain GGUF in one sentence." --once
```

The full-screen chat is used when both stdin and stdout are terminals. It has a
conversation viewport, a multi-line input box, a status bar and streaming
answers. Use `--no-tui` to force the plain line-based chat, for example in a
terminal multiplexer that does not support alternate screens.

## Commands

```bash
fastloadai scan
fastloadai scan --json
fastloadai suggest
fastloadai suggest --json
fastloadai download Qwen/Qwen2.5-0.5B-Instruct-GGUF
fastloadai download Qwen/Qwen2.5-0.5B-Instruct-GGUF --dry-run
fastloadai download Qwen/Qwen2.5-0.5B-Instruct-GGUF --prefer quality
fastloadai download Qwen/Qwen2.5-7B-Instruct-GGUF --workers 8
fastloadai models
fastloadai models --json
fastloadai plan ./model.gguf --output config.json
fastloadai optimize qwen2.5-0.5b --profile speed
fastloadai optimize ./model.gguf --profile memory --kv-cache q4_0
fastloadai optimize ./model.gguf --kv-cache-k q8_0 --kv-cache-v f16 --cache-ram 2048
fastloadai optimize ./model.gguf --rope-scaling yarn --rope-scale 2 --yarn-orig-ctx 32768
fastloadai optimize ./model.gguf --swa-checkpoints 8 --checkpoint-min-step 2048
fastloadai optimize ./model.gguf --cpu-range 0-7 --no-repack --no-warmup
fastloadai doctor
fastloadai run
fastloadai run qwen2.5-0.5b
fastloadai run qwen2.5-0.5b --no-tui
fastloadai run Qwen/Qwen2.5-0.5B-Instruct-GGUF
fastloadai run --prompt "Introduce yourself." --once
fastloadai run --low-memory --ctx-size 1024
fastloadai run --ctx-size 8192 --threads 6 --temperature 0.4
fastloadai run qwen2.5-0.5b --profile speed --flash-attn on --kv-cache q8_0
fastloadai run qwen2.5-0.5b --speculative --no-tui
fastloadai run qwen2.5-0.5b --draft-model ./draft.gguf --draft-max 4
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
targets 8 bit. `--file` selects an exact file or shard. `--workers` sets the
parallel download width; `0` (default) auto-tunes it from the CPU count, with a
minimum of 4 and a maximum of 16, never more than the shard count. `--dry-run`
shows the choice, worker count and memory fit without downloading. Sharded
models are downloaded together and verified against the API size. If nothing
fits, the command refuses unless `--force` is set.

`models` lists downloaded GGUF files, grouped by shard, with size,
quantization and fit. `--json` is available for scripts.

`plan` writes a runtime config for this machine. Pass a local GGUF path to
include its size and fit. Optimization flags are accepted here too.

`optimize` inspects a model, detects whether it is a Mixture of Experts model,
and writes the best `llama.cpp` flags for the chosen profile. It is the explicit
way to inspect or change the runtime plan without starting a server. Use
`--json` to print the resulting config.

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
- `--profile auto|speed|balanced|memory|quality` selects an optimization plan.
- `--flash-attn`, `--kv-cache`, `--kv-cache-k`, `--kv-cache-v`, `--speculative` and `--draft-model` tune inference.
- `--kv-unified`/`--no-kv-unified`, `--cache-ram`, `--slot-save-path` tune the KV and prompt cache.
- `--rope-scaling`, `--rope-scale`, `--yarn-orig-ctx`, `--swa-checkpoints` and `--checkpoint-min-step` extend context and checkpointing.
- `--cpu-mask`/`--cpu-range`, `--no-repack` and `--no-warmup` tune CPU affinity and startup.
- `--cpu-moe`, `--n-cpu-moe`, `--moe-cache-mib` and `--override-tensor` tune MoE.
- `--batch-size`, `--ubatch-size`, `--threads-batch`, `--parallel` and `--no-cont-batching` tune batching.
- `--load-mode`, `--mlock`, `--no-mmap`, `--lazy-mode`, `--no-kv-offload` tune memory.
- `--cache-reuse`, `--context-shift`, `--swa-full`, `--numa`, `--split-mode`, `--tensor-split`, `--priority` and `--poll` are advanced knobs.
- `--token`, `--revision`, `--force` and `--workers` work like `download`.
- `--no-tui` uses the plain line-based chat instead of the full-screen UI.
- `--no-color` disables ANSI colour for logs and CI.

`quantize` runs `llama-quantize` on an existing GGUF file.

## Chat keys

| Key | Action |
| --- | --- |
| `Enter` | Send the message |
| `Ctrl+J` | Insert a newline without sending |
| `↑` / `↓` | Browse input history |
| `←` / `→` | Move the cursor |
| `Home` / `End` | Jump to the start or end of the line |
| `PgUp` / `PgDn` | Scroll the conversation |
| `Tab` | Complete a slash command |
| `Ctrl+C` | Cancel while streaming, or exit when idle |
| `Ctrl+D` | Exit when the input is empty |

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
other applications and to container limits. Hardware probes, Hugging Face
metadata and runtime checks run in parallel, and `run` reuses one hardware scan
for both download planning and chat setup. CPU threads use every logical CPU
visible to the process, including cgroup limits, capped at 32; override with
`--threads`. The context starts at 1K-8K
depending on free memory and doubles when the model leaves at least 4 GB of
headroom, capped at 8K. `mmap` lets `llama.cpp` map weights and load pages on
demand. The estimate is conservative, not a guarantee against running out of
memory.

GPU offload is enabled for NVIDIA GPUs when the model fits 60% of VRAM, and for
Apple Silicon unified memory when the model fits the budget. AMD and Intel GPUs
run on CPU, which is safe and still works.

## Optimization profiles and llama.cpp flags

FastLoadAI writes one config with all runtime settings. `optimize`, `plan`,
`download` and `run` share the same flags, and `run` reuses the profile saved in
the config unless a flag overrides it.

| Profile | Goal | Typical choices |
| --- | --- | --- |
| `auto` | Pick from the hardware and model fit | `balanced`, or `memory` when RAM is tight |
| `speed` | Fastest short answers | Flash Attention on, F16 KV, large batches, KV reuse, context shift, no warmup |
| `balanced` | Good default | Flash Attention auto, Q8_0 KV for 8K+ context, KV reuse |
| `memory` | Smallest RAM/VRAM footprint | Q4_0 KV, 128-token batches, half the batch threads, no KV offload, prompt cache off, no repack |
| `quality` | Least numerical loss | F16 KV, no speculative decoding, conservative batching |

The optimization layer covers these techniques:

- **MoE CPU offload.** For Mixture of Experts models, attention and shared
  weights stay on the GPU while experts stay in system RAM. FastLoadAI detects
  common MoE names (`MoE`, `Mixtral`, `Qwen3-30B-A3B`, `Llama-4-16E`,
  `DeepSeek-V3`, `gpt-oss`, `OLMoE`, and similar) and sets `--cpu-moe`.
  Use `--n-cpu-moe N` to keep only the first N layers' experts on the CPU, or
  `--override-tensor` for an exact llama.cpp placement rule.
- **MoE expert cache.** `--moe-cache-mib` keeps the hottest experts in VRAM for
  decode. `--moe-cache-cpu-share` lets the CPU compute a share of cache misses
  instead of uploading them. This is the `-cmoe`/`--moe-cache-mib` path from
  llama.cpp's MoE cache documentation.
- **Flash Attention.** `--flash-attn on|off|auto` selects the fused attention
  kernel. It is on for `speed` and `memory`, and automatic for `balanced` and
  `quality`.
- **KV cache quantization.** `--kv-cache` sets both `--cache-type-k` and
  `--cache-type-v`. Use `--kv-cache-k` and `--kv-cache-v` independently, for
  example `--kv-cache-k q8_0 --kv-cache-v f16` to save memory on keys while
  keeping values at full precision. Common values are `f16`, `bf16`, `q8_0`,
  `q5_1`, `q4_0` and `iq4_nl`; support depends on the llama.cpp build.
- **KV buffer and prompt cache.** `--kv-unified` shares one KV buffer across
  slots (`-kvu`), while `--no-kv-unified` keeps one buffer per slot.
  `--cache-ram` limits the in-RAM prompt cache in MiB (`-1` unlimited, `0`
  disabled); the memory profile and `--low-memory` disable it. `--slot-save-path`
  persists slot KV state between sessions.
- **Context extension.** `--rope-scaling none|linear|yarn`, `--rope-scale N`
  and `--yarn-orig-ctx N` extend the context beyond the GGUF training size.
  `--swa-checkpoints N` sets context/SWA checkpoints per slot, and
  `--checkpoint-min-step N` sets their minimum spacing (`0` = no minimum).
- **CPU affinity and startup.** On hybrid Intel CPUs (and Apple Silicon for
  reporting), FastLoadAI detects performance and efficiency cores. The
  `speed`, `balanced` and `quality` profiles pin llama.cpp to the performance
  cores with `--cpu-mask` and cap batch threads to their count. Override with
  `--cpu-mask` or `--cpu-range`. `--no-repack` disables runtime weight repacking
  when RAM is tight; `--no-warmup` skips the empty startup run and is automatic
  in the `speed` profile and with `--once`. Use `--repack` or `--warmup` to
  force the opposite when a saved config disabled them.
- **Speculative decoding.** `--speculative` enables ngram lookup decoding with
  no draft model. `--draft-model` loads a smaller draft GGUF and enables
  `draft-simple`. `--spec-type` selects any llama.cpp type, including
  `draft-eagle3`, `draft-mtp`, `ngram-simple` and `ngram-map-k`.
  `--draft-max`, `--draft-min`, `--draft-p-min` and `--draft-gpu-layers` tune
  the draft.
- **Batching and slots.** `--batch-size`, `--ubatch-size`, `--parallel` and
  `--no-cont-batching` map directly to llama.cpp. The memory profile uses small
  batches; speed and balanced use large ones.
- **Thread tuning.** Decode threads use physical/logical CPU limits from the
  hardware scan. `--threads-batch` sets prompt-processing threads separately.
- **Load and memory mode.** `--load-mode auto|none|mmap|mlock|mmap+mlock|dio`
  controls weight loading. `--no-mmap` is an alias for `--load-mode none`;
  `--mlock` is an alias for `--load-mode mmap+mlock`. `--lazy-mode` enables
  on-demand tensor reads, and `--no-kv-offload` keeps the KV cache on the CPU.
- **KV reuse and long chats.** `--cache-reuse N` reuses matching prompt chunks
  through KV shifting. `--context-shift` keeps generation going when the
  context fills. `--swa-full` uses the full-size sliding-window cache.
- **Multi-GPU and NUMA.** `--split-mode none|layer|row|tensor`,
  `--tensor-split 3,1` and `--numa distribute|isolate|numactl` expose llama.cpp
  placement controls.
- **Scheduling.** `--priority` maps to `--prio`; `--poll` maps to `--poll`.
  `--no-op-offload` disables host tensor operation offload when a backend has a
  bug with it.

Only flags that differ from llama.cpp defaults are written to the command line.
For example, F16 KV cache and `--flash-attn auto` are omitted, while
`--flash-attn on`, `--cache-type-k q8_0`, `--kv-unified`, `--cache-ram 0`,
`--cpu-moe` and `--spec-type ngram-simple` are passed. Explicit `0` values for
`--cache-ram`, `--swa-checkpoints` and `--checkpoint-min-step` are preserved.

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
is not automated. MoE models work when the installed `llama.cpp` build supports
the architecture; FastLoadAI adds the MoE placement flags but cannot add kernel
support that the build does not have. Optimization flags are forwarded to
`llama-server`, so an older build may reject a newer flag. Check
`llama-server --help` or disable the flag with an explicit profile.

GPU detection covers NVIDIA through `nvidia-smi` and Apple Silicon through
Metal. Other GPUs are treated as CPU-only.

## Verified with real models

The Go build was tested end to end on a 4-vCPU, 15.6 GB CPU-only container with
the official `llama.cpp` b12613 build:

| Model | Quant | Context | Verified result |
| --- | --- | --- | --- |
| Qwen2.5-0.5B-Instruct | Q4_K_M | 8192 | ~4-5 tok/s (short reply) |
| Qwen2.5-0.5B-Instruct, `speed` profile | Q4_K_M | 8192 | Flash Attention + KV reuse |
| Qwen2.5-0.5B-Instruct, `balanced` profile | Q4_K_M | 8192 | Q8_0 KV cache |
| Qwen2.5-0.5B-Instruct, `memory` profile | Q4_K_M | 4096 | Q4_0 KV cache, 128-token batches, prompt cache off, no repack |
| Qwen2.5-0.5B-Instruct, ngram speculative | Q4_K_M | 8192 | `--spec-type ngram-simple` |
| Qwen2.5-0.5B-Instruct, draft model | Q4_K_M | 8192 | same file as draft, `draft-simple` |
| Qwen2.5-0.5B-Instruct, split KV | Q4_K_M | 8192 | K `q8_0`, V `f16`, unified KV, 2 GiB prompt cache |
| Qwen2.5-0.5B-Instruct, per-slot KV | Q4_K_M | 8192 | `--parallel 2`, `--no-kv-unified`, 512 MiB prompt cache |
| Qwen2.5-0.5B-Instruct, RoPE/SWA | Q4_K_M | 8192 | `--rope-scaling yarn --rope-scale 2`, SWA checkpoints 8/2048 |
| Qwen2.5-0.5B-Instruct, startup flags | Q4_K_M | 8192 | `--no-repack`, `--no-warmup`, `--cpu-range 0-3` |

Speeds are short-generation estimates and vary with prompt length, threads and
CPU. The optimization runs above were verified with the real Qwen2.5-0.5B GGUF
and the official `llama.cpp` b12613 build. MoE detection and placement are
covered by unit tests; the test container had no large MoE GGUF to load.

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
internal/optimization/  profiles, MoE detection and llama.cpp flag selection
internal/tui/       terminal UI, colour, tables, spinner and stats
```
