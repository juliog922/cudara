"""
Integration Tests for Cudara API
================================
Tests that require a running server or mock services.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_manager():
    """Mock ModelManager for testing."""
    with patch("src.cudara.main.ModelManager") as MockManager:
        manager = MagicMock()
        manager.get_allowed_models.return_value = {
            "test/model": MagicMock(
                description="Test model",
                task="text-generation",
                backend="transformers",
                architecture="AutoModelForCausalLM",
                quantization=MagicMock(bits=4),
                generation_defaults={},
            )
        }
        manager.get_registry.return_value = {
            "test/model": MagicMock(status=MagicMock(value="ready"), local_path="/models/test")
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
def mock_engine():
    """Mock InferenceEngine for testing."""
    with patch("src.cudara.main.InferenceEngine") as MockEngine:
        engine = MagicMock()
        engine.active_model_id = None
        engine.generate.return_value = {
            "model": "test/model",
            "created_at": "2025-01-01T00:00:00Z",
            "response": "Hello!",
            "done": True,
            "total_duration": 1000000,
            "eval_count": 5,
        }
        engine.chat.return_value = {
            "model": "test/model",
            "created_at": "2025-01-01T00:00:00Z",
            "response": "Hello!",
            "done": True,
            "total_duration": 1000000,
            "eval_count": 5,
        }
        engine.embeddings.return_value = {
            "model": "test/embedding",
            "embeddings": [[0.1, 0.2, 0.3]],
            "total_duration": 500000,
        }
        MockEngine.return_value = engine
        yield engine


@pytest.fixture
def client(mock_manager, mock_engine):
    """Create test client with mocked dependencies."""
    # Import after mocking
    with patch("src.cudara.main.manager", mock_manager):
        with patch("src.cudara.main.engine", mock_engine):
            from src.cudara.main import app

            with TestClient(app) as client:
                yield client


@pytest.mark.integration
class TestHealthEndpoint:
    """Tests for health endpoint."""

    def test_health_check(self, client):
        """Test health endpoint returns OK."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_root_endpoint(self, client):
        """Test root endpoint returns health."""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


@pytest.mark.integration
class TestModelsEndpoint:
    """Tests for model management endpoints."""

    def test_list_models(self, client):
        """Test listing models."""
        response = client.get("/api/tags")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_show_model(self, client):
        """Test showing model details."""
        response = client.post("/api/show", json={"name": "test/model"})
        assert response.status_code == 200
        data = response.json()
        assert "details" in data

    def test_show_model_not_found(self, client, mock_manager):
        """Test showing non-existent model."""
        mock_manager.get_model_info.return_value = None
        response = client.post("/api/show", json={"name": "nonexistent/model"})
        assert response.status_code == 404

    def test_show_model_missing_name(self, client):
        """Test show endpoint with missing name."""
        response = client.post("/api/show", json={})
        # FastAPI returns 422 Unprocessable Entity for missing required fields
        assert response.status_code == 422

    def test_pull_model_not_allowed(self, client, mock_manager):
        """Test pulling non-allowed model."""
        mock_manager.get_allowed_models.return_value = {}
        response = client.post("/api/pull", json={"name": "not/allowed"})
        assert response.status_code == 403

    def test_delete_model(self, client):
        """Test deleting a model."""
        response = client.request("DELETE", "/api/delete", json={"name": "test/model"})
        assert response.status_code == 200
        assert response.json()["status"] == "deleted"


@pytest.mark.integration
class TestGenerateEndpoint:
    """Tests for generate endpoint."""

    def test_generate_text(self, client, mock_engine):
        """Test text generation."""
        response = client.post("/api/generate", json={"model": "test/model", "prompt": "Hello"})
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert data["done"]
        mock_engine.generate.assert_called_once()

    def test_generate_with_system(self, client, mock_engine):
        """Test generation with system prompt."""
        response = client.post(
            "/api/generate",
            json={"model": "test/model", "prompt": "Hello", "system": "You are helpful"},
        )
        assert response.status_code == 200

    def test_generate_with_options(self, client, mock_engine):
        """Test generation with options."""
        response = client.post(
            "/api/generate",
            json={"model": "test/model", "prompt": "Hello", "options": {"temperature": 0.5}},
        )
        assert response.status_code == 200


@pytest.mark.integration
class TestChatEndpoint:
    """Tests for chat endpoint."""

    def test_chat_simple(self, client, mock_engine):
        """Test simple chat."""
        response = client.post(
            "/api/chat",
            json={"model": "test/model", "messages": [{"role": "user", "content": "Hello"}]},
        )
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"]["role"] == "assistant"

    def test_chat_multi_turn(self, client, mock_engine):
        """Test multi-turn chat."""
        response = client.post(
            "/api/chat",
            json={
                "model": "test/model",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi!"},
                    {"role": "user", "content": "How are you?"},
                ],
            },
        )
        assert response.status_code == 200


@pytest.mark.integration
class TestEmbeddingsEndpoint:
    """Tests for embeddings endpoint."""

    def test_embeddings_single(self, client, mock_engine):
        """Test single text embedding."""
        response = client.post(
            "/api/embeddings", json={"model": "test/embedding", "input": "Hello world"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "embeddings" in data
        mock_engine.embeddings.assert_called_once()

    def test_embeddings_multiple(self, client, mock_engine):
        """Test multiple text embeddings."""
        response = client.post(
            "/api/embeddings", json={"model": "test/embedding", "input": ["Hello", "World"]}
        )
        assert response.status_code == 200

    def test_embed_alias(self, client, mock_engine):
        """Test /api/embed alias endpoint."""
        response = client.post("/api/embed", json={"model": "test/embedding", "input": "Hello"})
        assert response.status_code == 200

    def test_embeddings_rerank_scores_shape_is_allowed(self, client, mock_engine):
        """Option 2: server may return 1D vectors (scores) for reranker models via /api/embeddings."""

        mock_engine.embeddings.return_value = {
            "model": "test/reranker",
            "embeddings": [[0.91], [0.12]],
            "total_duration": 500000,
        }

        sep = "\u241e"
        response = client.post(
            "/api/embeddings",
            json={"model": "test/reranker", "input": [f"q{sep}doc1", f"q{sep}doc2"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["embeddings"] == [[0.91], [0.12]]


@pytest.mark.integration
class TestLegacyEndpoints:
    """Tests for legacy endpoints."""

    def test_available_models_legacy(self, client):
        """Test legacy available-models endpoint."""
        response = client.get("/available-models")
        assert response.status_code == 200

    def test_models_legacy(self, client):
        """Test legacy models endpoint."""
        response = client.get("/models")
        assert response.status_code == 200


@pytest.mark.integration
class TestErrorResponses:
    """Tests for error response format."""

    def test_error_response_format(self, client, mock_engine):
        """Test error responses have correct format."""
        from src.cudara.main import AppError, ErrorCode

        mock_engine.generate.side_effect = AppError(
            "Test error", status_code=500, details={"code": ErrorCode.INFERENCE_ERROR}
        )

        response = client.post("/api/generate", json={"model": "test/model", "prompt": "Hello"})

        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert "code" in data["error"]
        assert "message" in data["error"]
