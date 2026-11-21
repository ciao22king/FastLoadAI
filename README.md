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

## Usage

```bash
fastloadai scan
fastloadai scan --json
fastloadai download bartowski/Llama-3.2-3B-Instruct-GGUF
fastloadai download Qwen/Qwen2.5-0.5B-Instruct-GGUF --prefer quality
fastloadai models
fastloadai doctor
fastloadai run
fastloadai run Llama-3.2-3B
fastloadai run --prompt "Introduce yourself." --once
fastloadai run --ctx-size 8192 --threads 6 --temperature 0.4
fastloadai plan ./model.gguf --output config.json
fastloadai quantize ./model.gguf ./model-Q4_K_M.gguf --type Q4_K_M
fastloadai --version
```

`scan` prints the detected hardware as a readable table. Add `--json` to get the raw JSON object instead.

`download` looks at the GGUF files in the repository and picks the best quantization that fits the estimated memory. `--prefer balanced` (default) targets a 4–5 bit K-quant, `speed` targets 4 bit, and `quality` targets 8 bit. Use `--file` to pick an explicit file. Gated or private repositories need `--token` or Hugging Face credentials in the environment. Sharded models are downloaded together.

`models` lists the GGUF files in the models folder, grouped by shard, with size and quantization. Add `--json` for scripts.

`doctor` checks Python, `llama-server`, `llama-quantize`, the Hugging Face extra, the config file and the model path. It exits non-zero when a required piece is missing.

`plan` writes a runtime config for the current machine. It prints a summary and saves the JSON file; use it when you already have a local GGUF.

`run` recalculates and saves the settings for the detected hardware, starts `llama-server` on `127.0.0.1` only, and opens the FastLoadAI chat with streaming responses and history. Omit the model name to reuse the model from the last `download`, or pass a name from the recorded repository or the `models` folder. If more than one file matches, pass the full GGUF path. `--prompt` sends a first message; `--once` answers it and exits, which also happens automatically when stdin is not a terminal. Tune a session with `--threads`, `--ctx-size`, `--gpu-layers`, `--temperature`, `--max-tokens` and `--system`. `--no-color` disables colors for logs and CI.

Inside the chat, `/help` lists the commands, `/stats` shows session speed and token estimates, `/context` shows how much of the window the conversation uses, `/retry` regenerates the last answer, `/clear` resets the conversation, and `/exit` (or `/quit`) closes the session and the server. History is trimmed by token budget, not by a fixed message count, so long chats stay inside the context window. The chat does not include the file-editing and shell tools of the OpenCode agent.

Each config records the detected specs and the recommended runtime settings: CPU threads, context size, mmap, and GPU offload when the file fits the VRAM budget. The context starts at 2K–8K depending on RAM and doubles when the model leaves at least 4 GB of headroom, capped at 8K. `mmap` lets `llama.cpp` map the weights and load pages on demand; the system can still slow down when physical memory is insufficient. The estimate is conservative, not a guarantee against running out of memory.

## Compatibility

FastLoadAI generates separate runtime configs and never modifies the original model files. `quantize` runs `llama-quantize` on an existing GGUF; converting Hugging Face checkpoints to GGUF is not automated. MoE models work only when the installed `llama.cpp` build supports the architecture. GPU detection currently covers NVIDIA through `nvidia-smi`; without a GPU the plan uses CPU and mmap.

## Verified with real models

End-to-end runs on a 4-vCPU, 15.6 GB CPU-only machine with the official `llama.cpp` b12608 build:

| Model | Quant | Context | Speed |
| --- | --- | --- | --- |
| Qwen2.5-0.5B-Instruct | Q4_K_M | 8192 | ~61 tok/s |
| Qwen2.5-0.5B-Instruct | Q8_0 | 8192 | ~52 tok/s |
| Llama-3.2-1B-Instruct | IQ4_XS | 8192 | ~11 tok/s |

Speeds are short-generation estimates and vary with prompt length, threads and CPU.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
