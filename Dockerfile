# -----------------------------------------------------------------------------
# Cudara — Lightweight CUDA Inference Server
# -----------------------------------------------------------------------------
ARG RUNTIME_IMAGE=nvidia/cuda:12.9.0-runtime-ubuntu22.04

FROM ${RUNTIME_IMAGE} AS runtime
ENV DEBIAN_FRONTEND=noninteractive

# 1. Runtime system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    libgomp1 \
    ca-certificates \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 2. Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# 3. UV configuration
ENV UV_PYTHON_DOWNLOADS=auto \
    UV_PYTHON=3.12 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH="/app/.venv/bin:${PATH}" \
    # Cudara env defaults (configurable at runtime)
    CUDARA_MODELS_DIR=/app/models \
    CUDARA_TEMP_DIR=/app/temp_uploads \
    CUDARA_IDLE_TIMEOUT=300

# 4. Dependency layer (cached unless lockfile changes)
COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-dev --extra vlm && \
    # Remove static archives to save space
    VENV_LIB=$(find .venv -name "site-packages" -type d | head -n 1) && \
    find ${VENV_LIB} -name "*.a" -delete 2>/dev/null || true

# 5. Application code
COPY src/ src/
COPY models.json ./

# 6. Model storage + registry (writable volume mount point)
RUN mkdir -p /app/models /app/temp_uploads && \
    echo "{}" > /app/registry.json && \
    chmod -R 777 /app/models /app/temp_uploads

# 7. Install the app itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-editable --extra vlm

# 8. Health check (uses the /health endpoint)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# 9. Use the installed entry point
CMD ["cudara", "serve", "--host", "0.0.0.0", "--port", "8000"]