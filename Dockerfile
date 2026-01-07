# syntax=docker/dockerfile:1.7-labs

ARG BUILD_IMAGE=nvidia/cuda:12.6.3-devel-ubuntu24.04
ARG RUNTIME_IMAGE=nvidia/cuda:12.6.3-base-ubuntu24.04
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.9.17

# Named stage (workaround for COPY --from not supporting variables)
FROM ${UV_IMAGE} AS uv

########################
# Builder
########################
FROM ${BUILD_IMAGE} AS builder
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-minimal \
    python3-venv \
    ca-certificates \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    git \
 && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /bin/

WORKDIR /app

ENV UV_NO_DEV=1 \
    UV_LINK_MODE=copy \
    CMAKE_ARGS="-DGGML_CUDA=on"

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project

COPY src/ src/
COPY models.json swagger.yaml index.html ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable

########################
# Runtime
########################
FROM ${RUNTIME_IMAGE} AS runtime
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    ca-certificates \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}"

# Copy the venv + app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/models.json /app/swagger.yaml /app/index.html /app/

EXPOSE 8000
CMD ["python3", "-m", "src.cudara.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
