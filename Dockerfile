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
ARG CUDA_ARCHS=86
ARG VERSION=dev
ARG COMMIT_SHA=unknown

# =============================================================================
# Runtime (CPU-only)
# =============================================================================
FROM ubuntu:24.04 AS runtime-cpu

ARG PYTHON_VERSION

ENV DEBIAN_FRONTEND=noninteractive     PYTHONUNBUFFERED=1     PYTHONDONTWRITEBYTECODE=1     UV_HTTP_TIMEOUT=600     UV_CONCURRENT_DOWNLOADS=2     UV_RETRIES=5

RUN apt-get update && apt-get install -y --no-install-recommends     python${PYTHON_VERSION}     python${PYTHON_VERSION}-venv     python3-pip     ffmpeg     libsndfile1     libgomp1     curl     && rm -rf /var/lib/apt/lists/*     && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3     && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install uv (fast package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY models.json .
COPY README.md .
# swagger.yaml may not exist in all contexts; copy if present
COPY swagger.yaml .

# CPU PyTorch wheels
RUN uv pip install --no-cache-dir     torch torchvision torchaudio     --index-url https://download.pytorch.org/whl/cpu

# Install runtime dependencies (from pyproject, excluding torch which is already installed)
RUN uv pip install --no-cache-dir     accelerate     bitsandbytes     "datasets>=4.4.1"     "fastapi>=0.124.4"     "huggingface-hub>=0.30.0,<1.0"     httpx     numpy     pillow     "pydantic>=2.12.3"     "python-multipart>=0.0.20"     "qwen-vl-utils>=0.0.14"     scipy     "transformers>=4.51.0"     "uvicorn>=0.38.0"

# Install cudara package (editable)
RUN uv pip install --no-cache-dir -e .

# Create runtime directories and empty registry
RUN mkdir -p models temp_uploads &&     echo "{}" > registry.json

# Health check (HTTP 200 => healthy; 503 => unhealthy)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3     CMD python -c "import httpx, sys; r=httpx.get('http://localhost:8000/health'); sys.exit(0 if r.status_code==200 else 1)" || exit 1

EXPOSE 8000

CMD ["cudara", "serve", "--host", "0.0.0.0", "--port", "8000"]


# =============================================================================
# Runtime (CUDA)  <-- default final image
# =============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04 AS runtime-cuda

ARG PYTHON_VERSION

ENV DEBIAN_FRONTEND=noninteractive     PYTHONUNBUFFERED=1     PYTHONDONTWRITEBYTECODE=1     UV_HTTP_TIMEOUT=600     UV_CONCURRENT_DOWNLOADS=2     UV_RETRIES=5

RUN apt-get update && apt-get install -y --no-install-recommends     python${PYTHON_VERSION}     python${PYTHON_VERSION}-venv     python3-pip     ffmpeg     libsndfile1     libgomp1     curl     && rm -rf /var/lib/apt/lists/*     && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3     && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY models.json .
COPY README.md .
COPY swagger.yaml .

# CUDA PyTorch (cu129)
RUN uv pip install --no-cache-dir     torch torchvision torchaudio     --index-url https://download.pytorch.org/whl/cu129

# Install runtime dependencies
RUN uv pip install --no-cache-dir     accelerate     bitsandbytes     "datasets>=4.4.1"     "fastapi>=0.124.4"     "huggingface-hub>=0.30.0,<1.0"     httpx     numpy     pillow     "pydantic>=2.12.3"     "python-multipart>=0.0.20"     "qwen-vl-utils>=0.0.14"     scipy     "transformers>=4.51.0"     "uvicorn>=0.38.0"

# Install cudara package (editable)
RUN uv pip install --no-cache-dir -e .

# Create runtime directories and empty registry
RUN mkdir -p models temp_uploads &&     echo "{}" > registry.json

# Health check (HTTP 200 => healthy; 503 => unhealthy)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3     CMD python -c "import httpx, sys; r=httpx.get('http://localhost:8000/health'); sys.exit(0 if r.status_code==200 else 1)" || exit 1

EXPOSE 8000

CMD ["cudara", "serve", "--host", "0.0.0.0", "--port", "8000"]
