"""API endpoint tests for Cudara."""

from __future__ import annotations

import base64
import json


# ===================================================================
# Model listing / info
# ===================================================================
class TestModelEndpoints:
    """Tests for /api/tags, /api/show, /api/ps, /api/version, /health."""

    def test_list_models(self, client):
        """GET /api/tags returns all 5 mock models."""
        r = client.get("/api/tags")
        assert r.status_code == 200
        models = r.json()["models"]
        assert len(models) == 5

    def test_show_model(self, client):
        """POST /api/show returns details for a known model."""
        r = client.post("/api/show", json={"model": "Qwen/Qwen2.5-3B-Instruct-AWQ"})
        assert r.status_code == 200
        body = r.json()
        assert "details" in body
        assert "completion" in body["capabilities"]

    def test_show_model_not_found(self, client):
        """POST /api/show returns 404 for unknown model."""
        r = client.post("/api/show", json={"model": "nonexistent/model"})
        assert r.status_code == 404
        assert "not found" in r.json()["error"].lower()

    def test_ps_empty(self, client):
        """GET /api/ps returns empty when no model is loaded."""
        r = client.get("/api/ps")
        assert r.status_code == 200
        assert r.json()["models"] == []

    def test_version(self, client):
        """GET /api/version returns a version string."""
        r = client.get("/api/version")
        assert r.status_code == 200
        assert "version" in r.json()

    def test_health(self, client):
        """GET /health returns ok."""
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


# ===================================================================
# Generate endpoint
# ===================================================================
class TestGenerate:
    """Tests for /api/generate."""

    def test_non_streaming(self, client):
        """Non-streaming generate returns a complete response."""
        r = client.post(
            "/api/generate",
            json={
                "model": "Qwen/Qwen2.5-3B-Instruct-AWQ",
                "prompt": "Write a haiku",
                "stream": False,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["model"] == "Qwen/Qwen2.5-3B-Instruct-AWQ"
        assert "response" in data
        assert data["done"] is True

    def test_streaming(self, client):
        """Streaming generate returns NDJSON lines."""
        with client.stream(
            "POST",
            "/api/generate",
            json={
                "model": "Qwen/Qwen2.5-3B-Instruct-AWQ",
                "prompt": "Hello",
                "stream": True,
            },
        ) as r:
            assert r.status_code == 200
            lines = list(r.iter_lines())
            assert len(lines) >= 1
            data = json.loads(lines[0])
            assert "response" in data

    def test_structured_json(self, client):
        """format='json' triggers JSON instruction injection."""
        r = client.post(
            "/api/generate",
            json={
                "model": "Qwen/Qwen2.5-3B-Instruct-AWQ",
                "prompt": "List 3 colors",
                "format": "json",
                "stream": False,
            },
        )
        assert r.status_code == 200
        assert r.json()["response"] == "Mocked JSON"

    def test_json_schema_format(self, client):
        """format as dict (JSON Schema) also triggers JSON mode."""
        r = client.post(
            "/api/generate",
            json={
                "model": "Qwen/Qwen2.5-3B-Instruct-AWQ",
                "prompt": "Give me structured data",
                "format": {"type": "object", "properties": {"name": {"type": "string"}}},
                "stream": False,
            },
        )
        assert r.status_code == 200
        assert r.json()["response"] == "Mocked JSON"

    def test_audio_transcription(self, client):
        """Audio data via images field with is_audio option."""
        dummy_audio = base64.b64encode(b"ID3...dummydata").decode()
        r = client.post(
            "/api/generate",
            json={
                "model": "openai/whisper-base",
                "prompt": "Transcribe this",
                "images": [dummy_audio],
                "options": {"is_audio": True},
                "stream": False,
            },
        )
        assert r.status_code == 200
        assert r.json()["response"] == "Mocked audio transcription."

    def test_model_not_found(self, client):
        """Unknown model returns 404."""
        r = client.post(
            "/api/generate",
            json={
                "model": "fake-model-123",
                "prompt": "hello",
                "stream": False,
            },
        )
        assert r.status_code == 404
        assert "not found" in r.json()["error"].lower()

    def test_system_prompt(self, client):
        """System prompt is accepted without error."""
        r = client.post(
            "/api/generate",
            json={
                "model": "Qwen/Qwen2.5-3B-Instruct-AWQ",
                "prompt": "Hi",
                "system": "You are a pirate.",
                "stream": False,
            },
        )
        assert r.status_code == 200
        assert r.json()["done"] is True

    def test_keep_alive_zero_unloads(self, client):
        """keep_alive=0 returns unload response."""
        r = client.post(
            "/api/generate",
            json={
                "model": "Qwen/Qwen2.5-3B-Instruct-AWQ",
                "keep_alive": 0,
            },
        )
        assert r.status_code == 200
        assert r.json()["done_reason"] == "unload"

    def test_empty_model_rejected(self, client):
        """Empty model string is rejected by validation."""
        r = client.post("/api/generate", json={"model": "", "prompt": "hi", "stream": False})
        assert r.status_code == 400


# ===================================================================
# Chat endpoint
# ===================================================================
class TestChat:
    """Tests for /api/chat."""

    def test_streaming(self, client):
        """Streaming chat returns NDJSON with message field."""
        with client.stream(
            "POST",
            "/api/chat",
            json={
                "model": "Qwen/Qwen2.5-3B-Instruct-AWQ",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        ) as r:
            assert r.status_code == 200
            lines = list(r.iter_lines())
            assert len(lines) > 0
            data = json.loads(lines[0])
            assert "message" in data

    def test_non_streaming(self, client):
        """Non-streaming chat returns a complete response."""
        r = client.post(
            "/api/chat",
            json={
                "model": "Qwen/Qwen2.5-3B-Instruct-AWQ",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["message"]["role"] == "assistant"
        assert data["done"] is True

    def test_missing_messages(self, client):
        """Missing messages field returns 400."""
        r = client.post("/api/chat", json={"model": "Qwen/Qwen2.5-3B-Instruct-AWQ"})
        assert r.status_code == 400
        assert "Invalid request" in r.json()["error"]

    def test_empty_messages(self, client):
        """Empty messages list returns 400."""
        r = client.post(
            "/api/chat",
            json={
                "model": "Qwen/Qwen2.5-3B-Instruct-AWQ",
                "messages": [],
            },
        )
        assert r.status_code == 400

    def test_invalid_role(self, client):
        """Invalid message role returns 400."""
        r = client.post(
            "/api/chat",
            json={
                "model": "Qwen/Qwen2.5-3B-Instruct-AWQ",
                "messages": [{"role": "villain", "content": "hello"}],
            },
        )
        assert r.status_code == 400


# ===================================================================
# Embed / Rerank
# ===================================================================
class TestEmbed:
    """Tests for /api/embed and /api/embeddings."""

    def test_single_text(self, client):
        """Embed a single string."""
        r = client.post(
            "/api/embed",
            json={
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "input": "Hello world",
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["embeddings"]) == 1
        assert len(data["embeddings"][0]) == 384

    def test_batch_texts(self, client):
        """Embed multiple strings."""
        r = client.post(
            "/api/embed",
            json={
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "input": ["Hello", "World"],
            },
        )
        assert r.status_code == 200
        assert len(r.json()["embeddings"]) == 2

    def test_custom_dimensions(self, client):
        """Custom dimensions truncates output."""
        r = client.post(
            "/api/embed",
            json={
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "input": "test",
                "dimensions": 64,
            },
        )
        assert r.status_code == 200
        assert len(r.json()["embeddings"][0]) == 64

    def test_rerank(self, client):
        """Rerank mode returns scores."""
        r = client.post(
            "/api/embed",
            json={
                "model": "BAAI/bge-reranker-v2-m3",
                "input": ["Query", "Doc 1", "Doc 2"],
                "options": {"is_rerank": True},
            },
        )
        assert r.status_code == 200
        assert "scores" in r.json()
        assert len(r.json()["scores"]) == 2

    def test_embeddings_alias(self, client):
        """POST /api/embeddings is an alias for /api/embed."""
        r = client.post(
            "/api/embeddings",
            json={
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "input": "test",
            },
        )
        assert r.status_code == 200
        assert "embeddings" in r.json()

    def test_empty_input_rejected(self, client):
        """Empty input string is rejected."""
        r = client.post(
            "/api/embed",
            json={
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "input": "",
            },
        )
        assert r.status_code == 400

    def test_empty_list_rejected(self, client):
        """Empty input list is rejected."""
        r = client.post(
            "/api/embed",
            json={
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "input": [],
            },
        )
        assert r.status_code == 400


# ===================================================================
# Delete
# ===================================================================
class TestDelete:
    """Tests for DELETE /api/delete."""

    def test_delete_model(self, client, monkeypatch):
        """Successful delete returns success status."""
        from cudara.main import manager as mgr

        monkeypatch.setattr(mgr, "delete_model", lambda model_id: None)
        r = client.request("DELETE", "/api/delete", json={"model": "Qwen/Qwen2.5-3B-Instruct-AWQ"})
        assert r.status_code == 200
        assert r.json()["status"] == "success"


# ===================================================================
# Stub endpoints
# ===================================================================
class TestStubs:
    """Tests for unimplemented endpoints."""

    def test_create_returns_501(self, client):
        """POST /api/create returns 501."""
        r = client.post("/api/create", json={})
        assert r.status_code == 501

    def test_copy_returns_501(self, client):
        """POST /api/copy returns 501."""
        r = client.post("/api/copy", json={})
        assert r.status_code == 501

    def test_push_returns_501(self, client):
        """POST /api/push returns 501."""
        r = client.post("/api/push", json={})
        assert r.status_code == 501
