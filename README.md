# Cudara

**Lightweight CUDA Inference Server with Ollama-Compatible API**

[![Docker](https://img.shields.io/badge/docker-ghcr.io%2Fjuliog922%2Fcudara-blue)](https://github.com/juliog922/cudara/pkgs/container/cudara)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Cudara is a self-hosted inference server for HuggingFace models. Run LLMs, Vision-Language Models, Embedding models, and Speech Recognition models on your GPU with an Ollama-compatible API.

## Features

- 🦙 **Ollama-Compatible API** - Works with existing Ollama clients
- 🖼️ **Vision-Language Models** - Image understanding, OCR, visual Q&A
- 💬 **Text Generation** - Chat and completion with any HuggingFace LLM
- 📊 **Embeddings** - Vector embeddings for RAG and semantic search
- 🎤 **Speech Recognition** - Transcribe audio with Whisper
- ⚡ **Quantization** - Automatic 4-bit quantization via BitsAndBytes
- 🔧 **GGUF Support** - Run GGUF models via llama.cpp

---

## Quick Start

### Using Docker (Recommended)

```bash
# Pull and run
docker run --gpus all -p 8000:8000 ghcr.io/juliog922/cudara:latest

# With persistent models
docker run --gpus all -p 8000:8000 \
  -v cudara_models:/app/models \
  ghcr.io/juliog922/cudara:latest
```

### Using uv (Development)

```bash
# Clone and install
git clone https://github.com/juliog922/cudara
cd cudara
uv sync

# Run server
uv run cudara serve
```

### Using the Client Library

```bash
pip install cudara-client
```

```python
from cudara_client import CudaraClient

client = CudaraClient("http://localhost:8000")
client.pull("Qwen/Qwen2.5-3B-Instruct")
response = client.chat("Qwen/Qwen2.5-3B-Instruct", "Hello!")
print(response.content)
```

---

## Project Structure

```
cudara/
├── src/cudara/
│   ├── __init__.py
│   ├── main.py              # FastAPI server
│   ├── cli.py               # CLI commands
│   ├── quantization.py      # BitsAndBytes quantization
│   └── image_processing.py  # VRAM-aware image processing
├── tests/
│   ├── test_unit.py         # Unit tests
│   └── test_integration.py  # Integration tests
├── .github/workflows/
│   ├── ci.yml               # Test on PR
│   └── docker-publish.yml   # Build & push to GHCR
├── models.json              # Model configurations
├── pyproject.toml           # Dependencies
├── Dockerfile               # Docker build
└── README.md
```

---

## CLI Usage

Cudara includes a CLI similar to Ollama:

```bash
# Start server
cudara serve --host 0.0.0.0 --port 8000

# List models
cudara list

# Pull a model
cudara pull Qwen/Qwen2.5-3B-Instruct

# Run inference
cudara run Qwen/Qwen2.5-3B-Instruct "Hello!"

# Interactive chat
cudara chat Qwen/Qwen2.5-3B-Instruct

# Show server status
cudara ps

# Delete model
cudara rm Qwen/Qwen2.5-3B-Instruct
```

---

## API Reference

### Ollama-Compatible Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/tags` | GET | List available models |
| `/api/show` | POST | Show model details |
| `/api/pull` | POST | Download a model |
| `/api/delete` | DELETE | Delete a model |
| `/api/generate` | POST | Generate text |
| `/api/chat` | POST | Chat completion |
| `/api/embeddings` | POST | Generate embeddings |

### Extended Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/transcribe` | POST | Transcribe audio |
| `/api/vision` | POST | Process image |
| `/health` | GET | Health check |

### Examples

```bash
# Chat
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-3B-Instruct", "messages": [{"role": "user", "content": "Hello!"}]}'

# Embeddings
curl -X POST http://localhost:8000/api/embeddings \
  -d '{"model": "sentence-transformers/all-MiniLM-L6-v2", "input": "Hello"}'

# Vision
curl -X POST http://localhost:8000/api/vision \
  -F "model=unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit" \
  -F "prompt=What is this?" -F "file=@image.jpg"

# Transcribe
curl -X POST http://localhost:8000/api/transcribe \
  -F "model=openai/whisper-small" -F "file=@audio.mp3"
```

---

## Adding Models

Edit `models.json` to add HuggingFace models:

### Text Generation

```json
"your-org/your-model": {
  "description": "Your model",
  "task": "text-generation",
  "architecture": "AutoModelForCausalLM",
  "dtype": "bfloat16",
  "quantization": {
    "enabled": true,
    "prequantize": true,
    "method": "bitsandbytes",
    "bits": 4,
    "category": "text_llm_medium"
  },
  "system_prompt": "You are helpful.",
  "generation_defaults": {"max_new_tokens": 512, "temperature": 0.7}
}
```

### Pre-quantized (Unsloth)

```json
"unsloth/Model-bnb-4bit": {
  "description": "Pre-quantized model",
  "task": "text-generation",
  "architecture": "AutoModelForCausalLM",
  "dtype": "bfloat16",
  "quantization": {"enabled": false, "notes": "Pre-quantized"}
}
```

### GGUF

```json
"org/model-GGUF": {
  "task": "text-generation",
  "backend": "gguf",
  "filename": "model-Q4_K_M.gguf",
  "parameters": {"n_gpu_layers": -1, "n_ctx": 8192}
}
```

---

## Development

### Run Tests

```bash
# All tests
uv run pytest

# Unit tests only
uv run pytest tests/test_unit.py -v

# Integration tests
uv run pytest tests/test_integration.py -v -m integration

# With coverage
uv run pytest --cov=src/cudara --cov-report=html
```

### Lint

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

### Build Docker

```bash
docker build -t cudara:latest .
docker run --gpus all -p 8000:8000 cudara:latest
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` | HuggingFace token for gated models | - |
| `CUDA_VISIBLE_DEVICES` | GPU selection | all |

---

## Requirements

- NVIDIA GPU with 8GB+ VRAM
- CUDA 12.1+
- Python 3.11+

---

## Cudara Client

Install the Python client:

```bash
pip install cudara-client
```

See [cudara-client](https://github.com/juliog922/cudara#cudara-client) for documentation.

---

## License

MIT