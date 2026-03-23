# -----------------------------------------------------------------------------
# GLOBAL ARGS
# -----------------------------------------------------------------------------
ARG RUNTIME_IMAGE=nvidia/cuda:12.9.0-runtime-ubuntu22.04

# -----------------------------------------------------------------------------
# RUNTIME STAGE (Single Stage Optimization)
# -----------------------------------------------------------------------------
FROM ${RUNTIME_IMAGE} AS runtime
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install Runtime Deps
# Kept ffmpeg for the Transformers audio transcription pipeline.
# Removed binutils because we no longer manually strip massive PyTorch binaries.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    libgomp1 \
    ca-certificates \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 2. Install 'uv' directly (Multi-stage copy strategy)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# 3. Configure UV
# - UV_LINK_MODE=copy: Vital for Docker caching across filesystems
# - UV_COMPILE_BYTECODE=1: Precompiles Python bytecode for much faster app startup
# - UV_PYTHON_DOWNLOADS=never: Use the system python we installed
ENV UV_PYTHON_DOWNLOADS=auto \
    UV_PYTHON=3.12 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH="/app/.venv/bin:${PATH}"

# 4. Copy Lockfiles
COPY pyproject.toml uv.lock ./

# 5. Install Dependencies DIRECTLY
# We use uv sync to fetch all dependencies.
# We removed the 'rm uv.lock' hack to ensure deterministic, locked builds.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-dev && \
    # Surgical cleanup: Remove static archives to save space
    VENV_LIB=$(find .venv -name "site-packages" -type d | head -n 1) && \
    find ${VENV_LIB} -name "*.a" -delete

# 6. Copy Application Code
COPY src/ src/
COPY models.json ./

RUN mkdir -p /app/models && \
    echo "{}" > /app/models/registry.json && \
    ln -s /app/models/registry.json /app/registry.json && \
    chmod -R 777 /app/models
    
# 7. Final Sync (Install the app itself)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-editable

EXPOSE 8000
CMD ["python3", "-m", "src.cudara.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]