"""
Unit Tests for Cudara.
======================
Fast, isolated tests for validating data models and CLI logic.
Refactored to remove dependencies on deleted modules.
"""

import json
from pathlib import Path
from unittest.mock import patch


class TestModels:
    """Tests for model configuration parsing."""

    def test_model_config_parsing(self, mock_models_json: Path) -> None:
        """Test that models.json structure is parsed correctly."""
        with open(mock_models_json) as f:
            models = json.load(f)

        assert "test-org/test-model" in models
        assert models["test-org/test-model"]["task"] == "text-generation"

    def test_model_config_quantization(self, mock_models_json: Path) -> None:
        """Verify parsing of the legacy quantization flag."""
        with open(mock_models_json) as f:
            models = json.load(f)

        model = models["test-org/test-model"]
        assert "quantization" in model
        assert model["quantization"]["load_in_4bit"] is False


class TestCLI:
    """Tests for CLI module structure and defaults."""

    def test_cli_import(self) -> None:
        """Ensure CLI components are importable."""
        from src.cudara.cli import cmd_list, cmd_pull, main

        assert callable(main)
        assert callable(cmd_list)
        assert callable(cmd_pull)

    def test_get_config_default(self, temp_dir: Path) -> None:
        """Test default config generation when no file exists."""
        from src.cudara.cli import DEFAULT_HOST, get_config

        with patch.object(Path, "exists", return_value=False):
            config = get_config()
            assert config["host"] == DEFAULT_HOST


class TestAPIModels:
    """Tests for Pydantic data models used in the API."""

    def test_generate_request_model(self) -> None:
        """Validate GenerateRequest fields."""
        from src.cudara.main import GenerateRequest

        req = GenerateRequest(model="test-model", prompt="Hello", stream=False)
        assert req.model == "test-model"
        assert req.prompt == "Hello"
        assert req.options == {}

    def test_chat_message_model(self) -> None:
        """Validate ChatMessage fields."""
        from src.cudara.main import ChatMessage

        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.images is None

    def test_embedding_request_model(self) -> None:
        """Validate EmbeddingRequest fields."""
        from src.cudara.main import EmbeddingRequest

        req = EmbeddingRequest(model="test", input="Hello")
        assert req.model == "test"
        assert req.input == "Hello"

        req_list = EmbeddingRequest(model="test", input=["Hello", "World"])
        assert isinstance(req_list.input, list)


class TestErrorHandling:
    """Tests for internal exception classes."""

    def test_app_error(self) -> None:
        """Test AppError properties."""
        from src.cudara.main import AppError

        error = AppError("Test error", status_code=404, code="model_not_found")
        assert error.message == "Test error"
        assert error.status_code == 404
        assert error.code == "model_not_found"


class TestModelStatus:
    """Tests for ModelStatus enum."""

    def test_model_status_values(self) -> None:
        """Ensure enum values match API expectations."""
        from src.cudara.main import ModelStatus

        assert ModelStatus.DOWNLOADING.value == "downloading"
        assert ModelStatus.READY.value == "ready"
        assert ModelStatus.ERROR.value == "error"
