# =============================================================================
# Cudara - CUDA Inference Server
# =============================================================================
# Build: docker build -t cudara:latest .
# Run:   docker run --gpus all -p 8000:8000 -v cudara_models:/app/models cudara:latest
# =============================================================================

# Stage 1: Build llama-cpp-python with CUDA support
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
# software-properties-common: for add-apt-repository
# git, cmake, build-essential: for compiling llama.cpp
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    cmake \
    git \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Create virtual environment
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Compilation Flags
# We link against the system CUDA in this stage.
# In the next stage, we will "trick" the binary to find these libs in the PyTorch folder.
ARG CUDA_ARCHS="86"
ENV CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}"
ENV FORCE_CMAKE=1

# Install llama-cpp-python
RUN uv pip install --no-cache-dir llama-cpp-python==0.3.16

# =============================================================================
# Stage 2: Runtime image
# =============================================================================
FROM nvidia/cuda:12.9.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_HTTP_TIMEOUT=600 \
    UV_CONCURRENT_DOWNLOADS=2 \
    UV_RETRIES=5

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python

# Copy venv with pre-built llama-cpp-python
COPY --from=builder /opt/venv /opt/venv
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

# Install PyTorch with CUDA 12.9
RUN uv pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu129

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

# Default command - use CLI
CMD ["cudara", "serve", "--host", "0.0.0.0", "--port", "8000"]

# =============================================================================
# Alternative commands:
# =============================================================================
# Run server directly:
#   CMD ["uvicorn", "src.cudara.main:app", "--host", "0.0.0.0", "--port", "8000"]
#
# Run with workers:
#   CMD ["cudara", "serve", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
# =============================================================================