# syntax=docker/dockerfile:1

# -----------------------------------------------------------------------------
# GLOBAL ARGS
# -----------------------------------------------------------------------------
ARG RUNTIME_IMAGE=nvidia/cuda:12.6.3-base-ubuntu24.04

# -----------------------------------------------------------------------------
# RUNTIME STAGE (Single Stage Optimization)
# -----------------------------------------------------------------------------
FROM ${RUNTIME_IMAGE} AS runtime
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install Runtime Deps + 'binutils' (for stripping)
# We install binutils temporarily to strip libraries, then remove it.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-minimal \
    python3-venv \
    libsndfile1 \
    libgomp1 \
    ca-certificates \
    binutils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Install 'uv' directly (No builder stage needed for this)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# 3. Configure UV
# - UV_LINK_MODE=copy: Vital for Docker caching
# - UV_COMPILE_BYTECODE=0: Disabling this makes builds faster & images smaller
# - UV_PYTHON_DOWNLOADS=never: Use the system python we installed
ENV UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=python3 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=0 \
    # Use the pre-built CUDA wheels (Critical for speed)
    UV_EXTRA_INDEX_URL="https://abetlen.github.io/llama-cpp-python/whl/cu124" \
    PATH="/app/.venv/bin:${PATH}"

# 4. Copy Lockfiles
COPY pyproject.toml uv.lock ./

# 5. Install Dependencies DIRECTLY (The Speed Fix)
# We mount the cache so 'uv' can link files instantly.
# We remove 'uv.lock' first to force the wheel resolution (as discussed previously).
# We also run the 'strip' command in the same layer to reduce size immediately.
RUN --mount=type=cache,target=/root/.cache/uv \
    rm uv.lock && \
    uv sync --no-install-project --no-dev && \
    # SURGICAL CLEANUP IN RUNTIME
    # Remove heavy static libs and strip symbols to save space
    VENV_LIB=$(find .venv -name "site-packages" -type d | head -n 1) && \
    find ${VENV_LIB} -name "*.so" -exec strip --strip-unneeded {} + 2>/dev/null || true && \
    find ${VENV_LIB} -name "*.a" -delete && \
    find ${VENV_LIB} -name "__pycache__" -exec rm -rf {} + && \
    # Remove binutils to keep the final image clean
    apt-get remove -y binutils && \
    apt-get autoremove -y

# 6. Copy Application Code
COPY src/ src/
COPY models.json swagger.yaml index.html ./

# 7. Final Sync (Install the app itself)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-editable

EXPOSE 8000
CMD ["python3", "-m", "src.cudara.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
