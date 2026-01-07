"""
Integration Tests for Cudara API.
=================================
Verifies interaction between endpoints and the backend logic.
Includes support for async locks and robust mocking.
"""

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
        # Setup mock allowed models
        manager.get_allowed.return_value = {
            "test/model": MagicMock(
                description="Test model",
                task="text-generation",
                backend="transformers",
                generation_defaults={},
            ),
            "test/reranker": MagicMock(task="feature-extraction", backend="transformers"),
        }
        # Setup mock registry
        manager.get_registry.return_value = {
            "test/model": MagicMock(status=MagicMock(value="ready"), local_path="/models/test"),
            "test/reranker": MagicMock(
                status=MagicMock(value="ready"), local_path="/models/rerank"
            ),
        }
        # Setup mock model info response
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
    Important: Since endpoints are async, mocked methods need to work in async contexts
    if they were awaited, but here we wrap synchronous methods in async endpoints.
    """
    with patch("src.cudara.main.InferenceEngine") as MockEngine:
        engine = MagicMock()
        engine.active_id = None

        # Mock generation/chat output
        engine.chat.return_value = {
            "model": "test/model",
            "created_at": "2025-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": "Hello!"},
            "response": "Hello!",  # For generate mapping
            "done": True,
            "total_duration": 1000000,
            "eval_count": 5,
            "eval_duration": 1000000,
        }

        # Mock embedding output
        engine.embeddings.return_value = {
            "model": "test/embedding",
            "embeddings": [[0.1, 0.2, 0.3]],
            "total_duration": 500000,
        }

        MockEngine.return_value = engine
        yield engine


@pytest.fixture
def client(mock_manager: MagicMock, mock_engine: MagicMock) -> Generator[TestClient, None, None]:
    """
    Create a TestClient with patched internal dependencies.
    """
    with patch("src.cudara.main.manager", mock_manager):
        with patch("src.cudara.main.engine", mock_engine):
            from src.cudara.main import app

            with TestClient(app) as client:
                yield client


@pytest.mark.integration
class TestHealthEndpoint:
    """Tests for system health endpoints."""

    def test_health_check(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_root_endpoint(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


@pytest.mark.integration
class TestModelsEndpoint:
    """Tests for model management endpoints."""

    def test_list_models(self, client: TestClient) -> None:
        response = client.get("/api/tags")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_show_model(self, client: TestClient) -> None:
        response = client.post("/api/show", json={"name": "test/model"})
        assert response.status_code == 200
        assert "details" in response.json()

    def test_show_model_not_found(self, client: TestClient, mock_manager: MagicMock) -> None:
        mock_manager.get_model_info.return_value = None
        response = client.post("/api/show", json={"name": "nonexistent/model"})
        assert response.status_code == 404

    def test_pull_model_not_allowed(self, client: TestClient, mock_manager: MagicMock) -> None:
        mock_manager.get_allowed.return_value = {}
        response = client.post("/api/pull", json={"name": "not/allowed"})
        assert response.status_code == 403

    def test_delete_model(self, client: TestClient) -> None:
        response = client.request("DELETE", "/api/delete", json={"name": "test/model"})
        assert response.status_code == 200
        assert response.json()["status"] == "deleted"


@pytest.mark.integration
class TestInferenceEndpoints:
    """Tests for generation, chat, and embeddings."""

    def test_generate_text(self, client: TestClient, mock_engine: MagicMock) -> None:
        """Test basic text generation logic."""
        response = client.post("/api/generate", json={"model": "test/model", "prompt": "Hello"})
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        mock_engine.chat.assert_called_once()  # API now maps generate -> chat

    def test_chat_simple(self, client: TestClient, mock_engine: MagicMock) -> None:
        """Test a simple chat interaction."""
        response = client.post(
            "/api/chat",
            json={"model": "test/model", "messages": [{"role": "user", "content": "Hello"}]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["message"]["content"] == "Hello!"

    def test_embeddings_single(self, client: TestClient, mock_engine: MagicMock) -> None:
        """Test single string embedding."""
        response = client.post(
            "/api/embeddings", json={"model": "test/embedding", "input": "Hello world"}
        )
        assert response.status_code == 200
        mock_engine.embeddings.assert_called_once()

    def test_embeddings_rerank(self, client: TestClient, mock_engine: MagicMock) -> None:
        """Test reranking support passing through correctly."""
        # Setup mock to return scores
        mock_engine.embeddings.return_value = {
            "model": "test/reranker",
            "embeddings": [[0.9], [0.1]],
            "total_duration": 100,
        }

        sep = "\u241e"
        response = client.post(
            "/api/embeddings", json={"model": "test/reranker", "input": [f"q{sep}d1", f"q{sep}d2"]}
        )
        assert response.status_code == 200
        assert response.json()["embeddings"] == [[0.9], [0.1]]

    def test_error_response_format(self, client: TestClient, mock_engine: MagicMock) -> None:
        """Ensure exceptions result in consistent JSON."""
        from src.cudara.main import AppError

        # Mock side effect for the engine call
        mock_engine.chat.side_effect = AppError("Simulated Failure", 500, "simulated_error")

        response = client.post("/api/chat", json={"model": "test/model", "messages": []})

        assert response.status_code == 500
        data = response.json()
        assert data["error"]["code"] == "simulated_error"
        assert data["error"]["message"] == "Simulated Failure"
