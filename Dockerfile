# =============================================================================
# Cudara - Inference Server
# =============================================================================
# Default build produces the CUDA runtime image (runtime-cuda).
#
# Build (CUDA):
#   docker build -t cudara:cuda .
#
# Build (CPU-only):
#   docker build --target runtime-cpu -t cudara:cpu .
#
# Run (CUDA, requires NVIDIA Container Toolkit on the host):
#   docker run --gpus all -p 8000:8000 -v cudara_models:/app/models cudara:cuda
#
# Run (CPU):
#   docker run -p 8000:8000 -v cudara_models:/app/models cudara:cpu
# =============================================================================

ARG PYTHON_VERSION=3.12
ARG CUDA_VERSION=12.9.1

# =============================================================================
# Stage 1A: Build llama-cpp-python (CPU)
# =============================================================================
FROM ubuntu:24.04 AS builder-cpu

ARG PYTHON_VERSION

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Create virtual environment
RUN python${PYTHON_VERSION} -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Build llama-cpp-python without CUDA (safe on CPU-only CI runners)
ENV FORCE_CMAKE=1
ENV CMAKE_ARGS="-DGGML_CUDA=off -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF"

RUN uv pip install --no-cache-dir llama-cpp-python==0.3.16


# =============================================================================
# Stage 1B: Build llama-cpp-python (CUDA)
# =============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS builder-cuda

ARG PYTHON_VERSION

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Create virtual environment
RUN python${PYTHON_VERSION} -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Compilation flags
ARG CUDA_ARCHS="86"
ENV FORCE_CMAKE=1
ENV CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS} -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF"

# CUDA builds on CPU-only CI runners need a link-time libcuda (driver) presence.
# Use CUDA's stub libcuda to satisfy the linker.
RUN set -eux; \
    STUB_DIR="/usr/local/cuda/lib64/stubs"; \
    if [ -d /usr/local/cuda/targets/x86_64-linux/lib/stubs ]; then STUB_DIR="/usr/local/cuda/targets/x86_64-linux/lib/stubs"; fi; \
    ln -sf "${STUB_DIR}/libcuda.so" "${STUB_DIR}/libcuda.so.1"; \
    CMAKE_ARGS="${CMAKE_ARGS} -DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,${STUB_DIR} -DCMAKE_SHARED_LINKER_FLAGS=-Wl,-rpath-link,${STUB_DIR}" \
    uv pip install --no-cache-dir llama-cpp-python==0.3.16; \
    rm -f "${STUB_DIR}/libcuda.so.1"


# =============================================================================
# Stage 2A: Runtime (CPU)
# =============================================================================
FROM ubuntu:24.04 AS runtime-cpu

ARG PYTHON_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_HTTP_TIMEOUT=600 \
    UV_CONCURRENT_DOWNLOADS=2 \
    UV_RETRIES=5

RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python

# Copy venv with pre-built llama-cpp-python (CPU)
COPY --from=builder-cpu /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY src/ src/

# Install dependencies (llama-cpp-python already installed)
RUN uv pip install --no-cache-dir \
    accelerate \
    bitsandbytes \
    "datasets>=4.4.1" \
    "fastapi>=0.119.1" \
    "huggingface-hub>=0.30.0,<1.0" \
    numpy \
    pillow \
    "pydantic>=2.12.3" \
    "python-multipart>=0.0.20" \
    "qwen-vl-utils>=0.0.14" \
    scipy \
    "transformers>=4.51.0" \
    "uvicorn>=0.38.0" \
    httpx

# CPU PyTorch
RUN uv pip install --no-cache-dir torch torchvision torchaudio

# Install cudara package in editable mode
RUN uv pip install --no-cache-dir -e .

# Copy config files
COPY models.json .
COPY swagger.yaml .

# Create directories
RUN mkdir -p models temp_uploads && \
    echo "{}" > registry.json

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health')" || exit 1

EXPOSE 8000

CMD ["cudara", "serve", "--host", "0.0.0.0", "--port", "8000"]


# =============================================================================
# Stage 2B: Runtime (CUDA)  <-- default final image
# =============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04 AS runtime-cuda

ARG PYTHON_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_HTTP_TIMEOUT=600 \
    UV_CONCURRENT_DOWNLOADS=2 \
    UV_RETRIES=5

RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python

# Copy venv with pre-built llama-cpp-python (CUDA)
COPY --from=builder-cuda /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY src/ src/

# CUDA PyTorch (cu129)
RUN uv pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu129

# Install dependencies (llama-cpp-python already installed)
RUN uv pip install --no-cache-dir \
    accelerate \
    bitsandbytes \
    "datasets>=4.4.1" \
    "fastapi>=0.119.1" \
    "huggingface-hub>=0.30.0,<1.0" \
    numpy \
    pillow \
    "pydantic>=2.12.3" \
    "python-multipart>=0.0.20" \
    "qwen-vl-utils>=0.0.14" \
    scipy \
    "transformers>=4.51.0" \
    "uvicorn>=0.38.0" \
    httpx

# Install cudara package in editable mode
RUN uv pip install --no-cache-dir -e .

# Copy config files
COPY models.json .
COPY swagger.yaml .

# Create directories
RUN mkdir -p models temp_uploads && \
    echo "{}" > registry.json

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health')" || exit 1

EXPOSE 8000

CMD ["cudara", "serve", "--host", "0.0.0.0", "--port", "8000"]
