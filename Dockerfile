FROM nvidia/cuda:12.9.1-base-ubuntu24.04

# 1. Install Python & System Dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    curl \
    ca-certificates \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 3. Configure Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app"

WORKDIR /app

# 4. Install Dependencies
COPY pyproject.toml README.md ./

# Create venv
RUN uv venv --python /usr/bin/python3
# Install deps without the project first
RUN uv sync --no-dev --no-install-project --no-cache

# 5. Copy Source & Install Project
COPY src/ src/
COPY models.json swagger.yaml index.html ./

# Now install the project itself
RUN uv sync --no-dev --no-cache

# 6. Final Setup
EXPOSE 8000

# Invoke the CLI module directly via python to avoid PATH/Alias issues
CMD ["python3", "-m", "src.cudara.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]