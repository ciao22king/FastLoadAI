#!/bin/sh
set -eu

REPO_URL="https://github.com/ciao22king/FastLoadAI.git"
REF=${FASTLOADAI_REF:-main}
INSTALL_DIR=${FASTLOADAI_HOME:-"$HOME/.local/share/fastloadai"}
VENV_DIR=${FASTLOADAI_VENV:-"$HOME/.local/share/fastloadai-venv"}
BIN_DIR=${FASTLOADAI_BIN_DIR:-"$HOME/.local/bin"}
PYTHON_BIN=${PYTHON:-python3}

if ! command -v git >/dev/null 2>&1; then
    printf '%s\n' "Errore: git non è installato." >&2
    exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
    printf '%s\n' "Errore: curl non è installato." >&2
    exit 1
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    printf '%s\n' "Errore: Python 3 non è installato o PYTHON non è valido." >&2
    exit 1
fi

if [ -d "$INSTALL_DIR/.git" ]; then
    git -C "$INSTALL_DIR" fetch --depth 1 origin "$REF"
    git -C "$INSTALL_DIR" checkout "$REF"
    git -C "$INSTALL_DIR" merge --ff-only FETCH_HEAD
elif [ -e "$INSTALL_DIR" ]; then
    printf 'Errore: %s esiste e non è un clone FastLoadAI; non lo sovrascrivo.\n' "$INSTALL_DIR" >&2
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
    printf 'Errore: %s esiste già e non è un link simbolico; non lo sovrascrivo.\n' "$BIN_DIR/fastloadai" >&2
    exit 1
fi
ln -sfn "$VENV_DIR/bin/fastloadai" "$BIN_DIR/fastloadai"

printf '\nFastLoadAI installato. Avvia con: fastloadai --help\n'
case ":${PATH:-}:" in
    *":$BIN_DIR:"*) ;;
    *) printf 'Aggiungi al PATH: export PATH="%s:$PATH"\n' "$BIN_DIR" ;;
esac
if ! command -v llama-server >/dev/null 2>&1; then
    printf '%s\n' "Nota: installa llama.cpp e rendi llama-server disponibile nel PATH per usare fastloadai run."
fi