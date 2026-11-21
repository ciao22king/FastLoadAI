# FastLoadAI

A command-line tool that downloads Hugging Face GGUF models, tunes them to your hardware, and runs local inference with `llama.cpp`.

FastLoadAI detects the CPU, RAM and GPU of the machine, picks a quantization that fits, writes a runtime config, and opens a streaming chat through `llama-server`.

## Install

Requires Git, curl and Python 3.10 or newer. After the repository is published on GitHub, install the CLI with:

```bash
curl -fsSL https://raw.githubusercontent.com/ciao22king/FastLoadAI/main/install.sh | sh
```

The installer creates an isolated virtualenv in `~/.local/share/fastloadai-venv`, installs `huggingface_hub`, downloads the official CPU builds of `llama.cpp` (server, CLI and quantizer), and adds the commands to `~/.local/bin`. It supports Linux and macOS on x86_64 and ARM64. If `~/.local/bin` is not on your `PATH`, add it as the installer suggests.

The installer does not compile `llama.cpp` and does not need `sudo`; it installs a CPU build. For GPU or custom builds, install a suitable `llama.cpp` build manually.

## Docker

A Docker image is included for machines where you do not want to install Python or `llama.cpp`. It is CPU-only and builds for `linux/amd64` and `linux/arm64`.

```bash
docker build -t fastloadai .
docker volume create fastloadai-data
docker run --rm -it -v fastloadai-data:/data fastloadai run qwen2.5-0.5b
```

`run <model name>` detects the container limits, picks the best quantization for that model, downloads it and starts the chat. You do not need a separate download command when you name the model. The `/data` volume keeps models, the Hugging Face cache and `fastloadai-config.json` between runs. Add `-e HF_TOKEN=...` for gated repositories. `--memory=4g` limits the container; FastLoadAI reads the cgroup limit, so it chooses settings that fit that limit. Use `run --low-memory` on very small limits.

To pre-download without starting the chat:

```bash
docker run --rm -v fastloadai-data:/data fastloadai download Qwen/Qwen2.5-0.5B-Instruct-GGUF
```

The same workflow is available with Compose:

```bash
docker compose build
docker compose run --rm fastloadai run qwen2.5-0.5b
```

The image uses the CPU build of `llama.cpp`. For GPU acceleration, use the native installer with a suitable `llama.cpp` build.

## Usage

```bash
fastloadai scan
fastloadai scan --json
fastloadai suggest
fastloadai download bartowski/Llama-3.2-3B-Instruct-GGUF
fastloadai download Qwen/Qwen2.5-0.5B-Instruct-GGUF --prefer quality
fastloadai download Qwen/Qwen2.5-7B-Instruct-GGUF --dry-run
fastloadai models
fastloadai doctor
fastloadai run
fastloadai run qwen2.5-0.5b
fastloadai run Qwen/Qwen2.5-0.5B-Instruct-GGUF
fastloadai run Llama-3.2-3B
fastloadai run --prompt "Introduce yourself." --once
fastloadai run --low-memory --ctx-size 1024
fastloadai run --ctx-size 8192 --threads 6 --temperature 0.4
fastloadai plan ./model.gguf --output config.json
fastloadai quantize ./model.gguf ./model-Q4_K_M.gguf --type Q4_K_M
fastloadai --version
```

`scan` prints the detected hardware as a readable table: CPU threads, total and free RAM, free disk, GPU, and the estimated model budget. Add `--json` to get the raw JSON object instead.

`suggest` lists small models that fit this machine and marks the best recommendation for the available memory and CPU. It prints the exact `run` command for the recommendation, so on a low-spec laptop you can copy that one line and let FastLoadAI download and configure the model. Add `--json` for scripts.

`download` looks at the GGUF files in the repository and picks the best quantization that fits the estimated memory. `--prefer balanced` (default) targets a 4–5 bit K-quant, `speed` targets 4 bit, and `quality` targets 8 bit. Use `--file` to pick an explicit file. `--dry-run` shows the selected file and its memory fit without downloading. If no quantization fits, FastLoadAI refuses to download unless you pass `--force`; it also checks free disk space first. Sharded models are downloaded together. Gated or private repositories need `--token`, `HF_TOKEN`, or a logged-in Hugging Face session. Use `--revision` for a branch, tag or commit.

`models` lists the GGUF files in the models folder, grouped by shard, with size and quantization. Add `--json` for scripts.

`doctor` checks Python, RAM, disk, GPU, `llama-server`, `llama-quantize`, the Hugging Face extra, the config file, the model path and the model fit. It exits non-zero when a required piece is missing.

`plan` writes a runtime config for the current machine. It prints a summary and saves the JSON file; use it when you already have a local GGUF.

`run` recalculates and saves the settings for the detected hardware, starts `llama-server` on `127.0.0.1` only, and opens the FastLoadAI chat with streaming responses and history. Pass a short catalog name such as `qwen2.5-0.5b`, a Hugging Face repo ID, a name from the recorded repository, or a local GGUF path. If the model is not local, `run` downloads the best quantization for this machine and starts the chat in one command. Omit the model name to reuse the configured model; if there is no config, `run` asks for a model name. If more than one file matches, pass the full GGUF path. `--prefer` controls the quantization trade-off for auto-downloads, and `--token`, `--revision` and `--force` work like they do for `download`. `--prompt` sends a first message; `--once` answers it and exits, which also happens automatically when stdin is not a terminal. Tune a session with `--threads`, `--ctx-size`, `--gpu-layers`, `--temperature`, `--max-tokens` and `--system`. `--low-memory` forces a small context and small batches, which helps on 2–4 GB laptops. `--no-color` disables colors for logs and CI. If the model is larger than the estimated memory, FastLoadAI warns before starting and suggests a smaller quantization; if `llama-server` runs out of memory, the error includes the same advice.

Inside the chat, `/help` lists the commands, `/stats` shows session speed and token estimates, `/context` shows how much of the window the conversation uses, `/retry` regenerates the last answer, `/clear` resets the conversation, and `/exit` (or `/quit`) closes the session and the server. History is trimmed by token budget, not by a fixed message count, so long chats stay inside the context window. The chat does not include the file-editing and shell tools of the OpenCode agent.

Each config records the detected specs and the recommended runtime settings: CPU threads, context size, mmap, model fit, low-memory mode, and GPU offload when the file fits the VRAM budget. The budget uses currently available RAM, not total RAM, so it adapts to other apps already running. The context starts at 1K–8K depending on free memory and doubles when the model leaves at least 4 GB of headroom, capped at 8K. `mmap` lets `llama.cpp` map the weights and load pages on demand; the system can still slow down when physical memory is insufficient. The estimate is conservative, not a guarantee against running out of memory.

## Compatibility

FastLoadAI generates separate runtime configs and never modifies the original model files. `quantize` runs `llama-quantize` on an existing GGUF; converting Hugging Face checkpoints to GGUF is not automated. MoE models work only when the installed `llama.cpp` build supports the architecture.

GPU detection covers NVIDIA through `nvidia-smi` and Apple Silicon unified memory through Metal. AMD and Intel GPUs are treated as CPU-only, which is safe and still works. Without a GPU, the plan uses CPU threads and mmap; models that fit in free RAM run normally, and models that do not fit are flagged before download or before start.

## Verified with real models

End-to-end runs on a 4-vCPU, 15.6 GB CPU-only machine with the official `llama.cpp` b12608 build:

| Model | Quant | Context | Speed |
| --- | --- | --- | --- |
| SmolLM2-135M-Instruct | Q4_K_M | 8192 | ~100 tok/s |
| Qwen2.5-0.5B-Instruct | Q4_K_M | 8192 | ~61 tok/s |
| Qwen2.5-0.5B-Instruct | Q8_0 | 8192 | ~52 tok/s |
| Llama-3.2-1B-Instruct | IQ4_XS | 8192 | ~11 tok/s |

Speeds are short-generation estimates and vary with prompt length, threads and CPU.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
