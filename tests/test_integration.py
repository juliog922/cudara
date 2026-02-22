"""
Integration Tests for Cudara API.
=================================
Verifies interaction between endpoints and the backend logic.
Includes support for async locks and robust mocking.
"""

import json
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_manager() -> Generator[MagicMock, None, None]:
    """
    Mock the ModelManager to simulate model configurations and registry states.
    """
    with patch("src.cudara.main.ModelManager") as MockManager:
        manager = MagicMock()
        manager.get_allowed.return_value = {
            "test/model": MagicMock(
                description="Test model",
                task="text-generation",
                backend="transformers",
                generation_defaults={},
            ),
            "test/reranker": MagicMock(task="feature-extraction", backend="transformers"),
        }
        manager.get_registry.return_value = {
            "test/model": MagicMock(status=MagicMock(value="ready"), local_path="/models/test"),
            "test/reranker": MagicMock(status=MagicMock(value="ready"), local_path="/models/rerank"),
        }
        manager.get_model_info.return_value = {
            "modelfile": "# test",
            "parameters": "{}",
            "template": "",
            "details": {"format": "transformers"},
            "model_info": {"task": "text-generation", "status": "ready"},
        }
        MockManager.return_value = manager
        yield manager


@pytest.fixture
def mock_engine() -> Generator[MagicMock, None, None]:
    """
    Mock the InferenceEngine to simulate model outputs.
    Handles stream vs non-stream returns precisely as the backend expects.
    """
    with patch("src.cudara.main.InferenceEngine") as MockEngine:
        engine = MagicMock()
        engine.active_id = None

        def mock_chat_logic(model_id, messages, options, stream=False):
            if stream:
                # Return a synchronous generator yielding JSON strings (simulating Llama-cpp output)
                def gen():
                    yield (
                        json.dumps(
                            {
                                "model": model_id,
                                "created_at": "2025-01-01T00:00:00Z",
                                "message": {"content": "Hello"},
                                "done": False,
                            }
                        )
                        + "\n"
                    )

                    yield (
                        json.dumps(
                            {
                                "model": model_id,
                                "created_at": "2025-01-01T00:00:00Z",
                                "message": {"content": "!"},
                                "done": True,
                                "eval_count": 2,
                            }
                        )
                        + "\n"
                    )

                return gen()
            else:
                return {
                    "model": model_id,
                    "created_at": "2025-01-01T00:00:00Z",
                    "message": {"role": "assistant", "content": "Hello!"},
                    "response": "Hello!",
                    "done": True,
                    "total_duration": 1000000,
                    "eval_count": 5,
                    "eval_duration": 1000000,
                }

        engine.chat.side_effect = mock_chat_logic

        engine.embeddings.return_value = {
            "model": "test/embedding",
            "embeddings": [[0.1, 0.2, 0.3]],
            "total_duration": 500000,
        }

        MockEngine.return_value = engine
        yield engine


@pytest.fixture
def client(mock_manager: MagicMock, mock_engine: MagicMock) -> Generator[TestClient, None, None]:
    """Create a TestClient with patched internal dependencies."""
    with patch("src.cudara.main.manager", mock_manager):
        with patch("src.cudara.main.engine", mock_engine):
            from src.cudara.main import app

            with TestClient(app) as client:
                yield client


@pytest.mark.integration
class TestHealthEndpoint:
    """Tests for the API health check endpoints."""

    def test_health_check(self, client: TestClient) -> None:
        """Verify the /health endpoint returns a 200 OK status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_root_endpoint(self, client: TestClient) -> None:
        """Verify the root (/) endpoint acts as a health check."""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


@pytest.mark.integration
class TestModelsEndpoint:
    """Tests for model management and info endpoints."""

    def test_list_models(self, client: TestClient) -> None:
        """Verify the /api/tags endpoint lists available models."""
        response = client.get("/api/tags")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_show_model(self, client: TestClient) -> None:
        """Verify the /api/show endpoint returns details for an existing model."""
        response = client.post("/api/show", json={"name": "test/model"})
        assert response.status_code == 200
        assert "details" in response.json()

    def test_show_model_not_found(self, client: TestClient, mock_manager: MagicMock) -> None:
        """Verify the /api/show endpoint returns 404 for missing models."""
        mock_manager.get_model_info.return_value = None
        response = client.post("/api/show", json={"name": "nonexistent/model"})
        assert response.status_code == 404


@pytest.mark.integration
class TestInferenceEndpoints:
    """Tests for text generation, chat, and embedding endpoints."""

    def test_generate_text(self, client: TestClient, mock_engine: MagicMock) -> None:
        """Verify standard text generation returns expected structure."""
        response = client.post("/api/generate", json={"model": "test/model", "prompt": "Hello"})
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        mock_engine.chat.assert_called_once()

    def test_generate_text_stream(self, client: TestClient, mock_engine: MagicMock) -> None:
        """Verify streaming text generation yields NDJSON chunks."""
        response = client.post("/api/generate", json={"model": "test/model", "prompt": "Hello", "stream": True})
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/x-ndjson"

        # Verify chunks
        lines = [line for line in response.text.split("\n") if line.strip()]
        assert len(lines) == 2
        assert json.loads(lines[0])["response"] == "Hello"
        assert json.loads(lines[1])["done"] is True

    def test_chat_simple(self, client: TestClient, mock_engine: MagicMock) -> None:
        """Verify standard chat completion returns expected structure."""
        response = client.post(
            "/api/chat",
            json={"model": "test/model", "messages": [{"role": "user", "content": "Hello"}]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["message"]["content"] == "Hello!"

    def test_chat_stream(self, client: TestClient, mock_engine: MagicMock) -> None:
        """Verify streaming chat completion yields NDJSON chunks."""
        response = client.post(
            "/api/chat",
            json={"model": "test/model", "messages": [{"role": "user", "content": "Hello"}], "stream": True},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/x-ndjson"

        lines = [line for line in response.text.split("\n") if line.strip()]
        assert len(lines) == 2
        assert json.loads(lines[0])["message"]["content"] == "Hello"

    def test_embeddings_single(self, client: TestClient, mock_engine: MagicMock) -> None:
        """Verify the embeddings endpoint processes a single input string."""
        response = client.post("/api/embeddings", json={"model": "test/embedding", "input": "Hello world"})
        assert response.status_code == 200
        mock_engine.embeddings.assert_called_once()

    def test_error_response_format(self, client: TestClient, mock_engine: MagicMock) -> None:
        """Verify that internal AppErrors are formatted correctly as JSON."""
        from src.cudara.main import AppError

        mock_engine.chat.side_effect = AppError("Simulated Failure", 500, "simulated_error")

        response = client.post("/api/chat", json={"model": "test/model", "messages": []})

        assert response.status_code == 500
        data = response.json()
        assert data["error"]["code"] == "simulated_error"
        assert data["error"]["message"] == "Simulated Failure"
