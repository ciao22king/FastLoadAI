# syntax=docker/dockerfile:1

FROM python:3.12-slim

ARG LLAMA_VERSION=b12608
ARG TARGETARCH=amd64

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/.cache/huggingface \
    LD_LIBRARY_PATH=/opt/llama/current

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libgomp1 \
        libssl3 \
        libzstd1 \
        zlib1g \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .

RUN set -eu; \
    case "${TARGETARCH}" in \
        amd64) LLAMA_ARCH=x64 ;; \
        arm64) LLAMA_ARCH=arm64 ;; \
        *) echo "Unsupported architecture: ${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    curl -fL --retry 3 --retry-delay 2 \
        -o /tmp/llama.tar.gz \
        "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_VERSION}/llama-${LLAMA_VERSION}-bin-ubuntu-${LLAMA_ARCH}.tar.gz"; \
    mkdir -p /opt/llama; \
    tar -xzf /tmp/llama.tar.gz -C /opt/llama; \
    LLAMA_SERVER=$(find /opt/llama -type f -name llama-server -print -quit); \
    test -n "${LLAMA_SERVER}"; \
    LLAMA_DIR=$(dirname "${LLAMA_SERVER}"); \
    ln -sfn "${LLAMA_DIR}" /opt/llama/current; \
    for TOOL in llama-server llama-cli llama-quantize; do \
        if [ -f "${LLAMA_DIR}/${TOOL}" ]; then \
            ln -sf "${LLAMA_DIR}/${TOOL}" "/usr/local/bin/${TOOL}"; \
        fi; \
    done; \
    rm -f /tmp/llama.tar.gz

WORKDIR /data
VOLUME ["/data"]

ENTRYPOINT ["fastloadai"]
CMD ["--help"]
