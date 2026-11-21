#!/bin/sh
set -eu

REPO_URL="https://github.com/ciao22king/FastLoadAI.git"
REF=${FASTLOADAI_REF:-main}
INSTALL_DIR=${FASTLOADAI_HOME:-"$HOME/.local/share/fastloadai"}
VENV_DIR=${FASTLOADAI_VENV:-"$HOME/.local/share/fastloadai-venv"}
BIN_DIR=${FASTLOADAI_BIN_DIR:-"$HOME/.local/bin"}
LLAMA_ROOT=${FASTLOADAI_LLAMA_HOME:-"$HOME/.local/share/fastloadai-llama"}
PYTHON_BIN=${PYTHON:-python3}

if ! command -v git >/dev/null 2>&1; then
    printf '%s\n' "Error: git is not installed." >&2
    exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
    printf '%s\n' "Error: curl is not installed." >&2
    exit 1
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    printf '%s\n' "Error: Python 3 is not installed or PYTHON is invalid." >&2
    exit 1
fi

if [ -d "$INSTALL_DIR/.git" ]; then
    if [ -n "$(git -C "$INSTALL_DIR" status --porcelain)" ]; then
        printf 'Error: %s has local changes; refusing to overwrite them.\n' "$INSTALL_DIR" >&2
        exit 1
    fi
    git -C "$INSTALL_DIR" fetch --depth 1 origin "$REF"
    git -C "$INSTALL_DIR" checkout -B "$REF" FETCH_HEAD
elif [ -e "$INSTALL_DIR" ]; then
    printf 'Error: %s exists and is not a FastLoadAI clone; refusing to overwrite it.\n' "$INSTALL_DIR" >&2
    exit 1
else
    mkdir -p "$(dirname "$INSTALL_DIR")"
    git clone --depth 1 --branch "$REF" "$REPO_URL" "$INSTALL_DIR"
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
    mkdir -p "$(dirname "$VENV_DIR")"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade "$INSTALL_DIR[huggingface]"

mkdir -p "$BIN_DIR"
if [ -e "$BIN_DIR/fastloadai" ] && [ ! -L "$BIN_DIR/fastloadai" ]; then
    printf 'Error: %s already exists and is not a symlink; refusing to overwrite it.\n' "$BIN_DIR/fastloadai" >&2
    exit 1
fi
ln -sfn "$VENV_DIR/bin/fastloadai" "$BIN_DIR/fastloadai"

if ! command -v llama-server >/dev/null 2>&1 || ! command -v llama-quantize >/dev/null 2>&1; then
    case "$(uname -s)" in
        Linux) LLAMA_PLATFORM=ubuntu ;;
        Darwin) LLAMA_PLATFORM=macos ;;
        *)
            printf 'Error: automatic llama.cpp installation is not supported on %s.\n' "$(uname -s)" >&2
            exit 1
            ;;
    esac

    case "$(uname -m)" in
        x86_64|amd64) LLAMA_ARCH=x64 ;;
        aarch64|arm64) LLAMA_ARCH=arm64 ;;
        *)
            printf 'Error: architecture not supported by llama.cpp: %s.\n' "$(uname -m)" >&2
            exit 1
            ;;
    esac

    LLAMA_SUFFIX="-bin-$LLAMA_PLATFORM-$LLAMA_ARCH.tar.gz"
    LLAMA_ASSET_URL=$(curl -fsSL "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=20" | "$VENV_DIR/bin/python" -c '
import json
import sys

suffix = sys.argv[1]
for release in json.load(sys.stdin):
    for asset in release.get("assets", []):
        if asset["name"].startswith("llama-") and asset["name"].endswith(suffix):
            print(asset["browser_download_url"])
            raise SystemExit(0)
' "$LLAMA_SUFFIX")
    if [ -z "$LLAMA_ASSET_URL" ]; then
        printf 'Error: no llama.cpp binary found for %s/%s.\n' "$LLAMA_PLATFORM" "$LLAMA_ARCH" >&2
        exit 1
    fi

    TMP_DIR=$(mktemp -d)
    trap 'rm -rf "$TMP_DIR"' 0
    printf 'Downloading the llama.cpp CPU runtime for %s/%s...\n' "$LLAMA_PLATFORM" "$LLAMA_ARCH"
    if ! curl -fL --retry 2 -o "$TMP_DIR/llama.cpp.tar.gz" "$LLAMA_ASSET_URL"; then
        printf '%s\n' "Error: llama.cpp runtime download failed." >&2
        exit 1
    fi
    if ! tar -xzf "$TMP_DIR/llama.cpp.tar.gz" -C "$TMP_DIR"; then
        printf '%s\n' "Error: invalid llama.cpp archive." >&2
        exit 1
    fi

    LLAMA_SERVER_PATH=$(find "$TMP_DIR" -type f -name llama-server -print -quit)
    if [ -z "$LLAMA_SERVER_PATH" ]; then
        printf '%s\n' "Error: llama-server not found in the official package." >&2
        exit 1
    fi
    LLAMA_SOURCE_DIR=$(dirname "$LLAMA_SERVER_PATH")
    LLAMA_BUILD_DIR="$LLAMA_ROOT/$(basename "$LLAMA_SOURCE_DIR")"
    mkdir -p "$LLAMA_ROOT"
    if [ ! -x "$LLAMA_BUILD_DIR/llama-server" ]; then
        mv "$LLAMA_SOURCE_DIR" "$LLAMA_BUILD_DIR"
    fi

    for TOOL in llama-server llama-cli llama-quantize; do
        if ! command -v "$TOOL" >/dev/null 2>&1 && [ -x "$LLAMA_BUILD_DIR/$TOOL" ]; then
            if [ -e "$BIN_DIR/$TOOL" ] && [ ! -L "$BIN_DIR/$TOOL" ]; then
                printf 'Error: %s already exists and is not a symlink; refusing to overwrite it.\n' "$BIN_DIR/$TOOL" >&2
                exit 1
            fi
            ln -sfn "$LLAMA_BUILD_DIR/$TOOL" "$BIN_DIR/$TOOL"
        fi
    done
fi

printf '\nFastLoadAI installed. Start with: fastloadai --help\n'
case ":${PATH:-}:" in
    *":$BIN_DIR:"*) ;;
    *) printf 'Add to PATH: export PATH="%s:$PATH"\n' "$BIN_DIR" ;;
esac
if ! command -v llama-server >/dev/null 2>&1; then
    printf '%s\n' "Error: llama-server was not installed or is not on PATH." >&2
    exit 1
fi