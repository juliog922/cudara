import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from cudara.main import AppError, ModelConfig, ModelStatus, RegistryItem, app, engine, manager


@pytest.fixture
def client():
    """Provides a FastAPI TestClient for routing requests."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_registry_and_engine(monkeypatch):
    """Mocks the filesystem registry and the PyTorch InferenceEngine."""

    mock_allowed = {
        "Qwen/Qwen2.5-3B-Instruct-AWQ": ModelConfig(task="text-generation"),
        "Qwen/Qwen2.5-VL-3B-Instruct-AWQ": ModelConfig(task="image-to-text"),
        "openai/whisper-base": ModelConfig(task="automatic-speech-recognition"),
        "BAAI/bge-reranker-v2-m3": ModelConfig(task="text-classification"),
        "sentence-transformers/all-MiniLM-L6-v2": ModelConfig(task="feature-extraction"),
    }
    monkeypatch.setattr(manager, "get_allowed", lambda: mock_allowed)

    # Point to the current tests directory so Path.exists() evaluates to True
    fake_path = str(Path(__file__).parent.resolve())
    mock_registry = {k: RegistryItem(status=ModelStatus.READY, local_path=fake_path) for k in mock_allowed.keys()}
    monkeypatch.setattr(manager, "get_registry", lambda: mock_registry)

    # Mock the InferenceEngine
    monkeypatch.setattr(engine, "load", MagicMock())
    monkeypatch.setattr(engine, "unload", MagicMock())

    def mock_chat(model_id, messages, options, stream, is_chat, force_json):
        if model_id not in mock_allowed:
            raise AppError(f"Model '{model_id}' not found", 404)

        res = {
            "model": model_id,
            "created_at": "2026-03-21T18:00:00.000Z",
            "done": True,
            "done_reason": "stop",
            "total_duration": 1000,
            "load_duration": 100,
            "prompt_eval_count": 10,
            "eval_count": 5,
            "eval_duration": 500,
        }

        thinking = "Dummy thought process" if options.get("think") else None
        content = "Mocked JSON" if force_json else "Mocked response"

        if is_chat:
            res["message"] = {"role": "assistant", "content": content}
        else:
            res["response"] = content

        if thinking:
            res["thinking"] = thinking

        # FIX: Explicitly return a dict if not streaming!
        if stream:

            def stream_generator():
                yield json.dumps(res) + "\n"

            return stream_generator()
        return res

    monkeypatch.setattr(engine, "chat", mock_chat)

    def mock_embeddings(model_id, texts, truncate, dimensions=None):
        size = dimensions if dimensions else 384
        return {
            "model": model_id,
            "embeddings": [[0.1] * size for _ in texts],
            "total_duration": 100,
            "prompt_eval_count": 5,
        }

    monkeypatch.setattr(engine, "embeddings", mock_embeddings)

    def mock_rerank(model_id, query, docs):
        return {"model": model_id, "scores": [0.99, 0.5, 0.1][: len(docs)], "total_duration": 50}

    monkeypatch.setattr(engine, "rerank", mock_rerank)

    def mock_transcribe(model_id, path):
        return {"model": model_id, "text": "Mocked audio transcription.", "total_duration": 200}

    monkeypatch.setattr(engine, "transcribe", mock_transcribe)
