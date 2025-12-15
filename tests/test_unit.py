"""
Unit Tests for Cudara
=====================
Fast tests without external dependencies.
"""

import json
from pathlib import Path
from unittest.mock import patch


class TestImageProcessing:
    """Tests for image_processing module."""

    def test_vram_profile_calculation(self):
        """Test VRAM profile calculates optimal settings."""
        from src.cudara.image_processing import VRAMProfile

        profile = VRAMProfile(total_vram_gb=24, available_vram_gb=18, model_vram_gb=4)

        assert profile.min_pixels == 200704
        assert profile.max_pixels > profile.min_pixels
        assert profile.optimal_pixels > profile.min_pixels
        assert profile.optimal_pixels < profile.max_pixels
        assert len(profile.resolution_tiers) > 0

    def test_vram_presets_exist(self):
        """Test that common GPU presets are defined."""
        from src.cudara.image_processing import VRAM_PRESETS

        expected_gpus = ["rtx_4090_24gb", "l40s_48gb", "a100_80gb", "h100_80gb"]
        for gpu in expected_gpus:
            assert gpu in VRAM_PRESETS, f"Missing preset: {gpu}"

    def test_l40s_preset(self):
        """Test L40S GPU preset values."""
        from src.cudara.image_processing import VRAM_PRESETS

        l40s = VRAM_PRESETS["l40s_48gb"]
        assert l40s.total_vram_gb == 48
        assert l40s.available_vram_gb == 40
        assert l40s.model_vram_gb == 6

    def test_image_processing_config_defaults(self):
        """Test default image processing config."""
        from src.cudara.image_processing import ImageProcessingConfig

        config = ImageProcessingConfig()
        assert config.patch_size == 14
        assert config.merge_size == 2
        assert config.min_pixels == 200704
        assert config.max_retry_attempts == 3

    def test_adaptive_processor_resize_dims(self):
        """Test dimension calculation for resize."""
        from src.cudara.image_processing import AdaptiveImageProcessor

        proc = AdaptiveImageProcessor()

        # Test dimension calculation
        w, h = proc._calculate_resize_dims(1920, 1080, 500000, 200704, 28)
        assert w * h <= 500000
        assert w % 28 == 0
        assert h % 28 == 0

    def test_estimate_tokens(self):
        """Test token estimation."""
        from src.cudara.image_processing import AdaptiveImageProcessor

        proc = AdaptiveImageProcessor()
        tokens = proc.estimate_tokens(512, 512)
        assert tokens > 0
        assert isinstance(tokens, int)


class TestQuantization:
    """Tests for quantization module."""

    def test_detect_model_category_text(self):
        """Test model category detection for text models."""
        from src.cudara.quantization import ModelCategory, detect_model_category

        # New logic only uses 'task', ignoring model name heuristics
        cat = detect_model_category(task="text-generation")
        assert cat == ModelCategory.TEXT_LLM

    def test_detect_model_category_vlm(self):
        """Test model category detection for VLMs."""
        from src.cudara.quantization import ModelCategory, detect_model_category

        cat = detect_model_category(task="image-to-text")
        assert cat == ModelCategory.VLM

    def test_detect_model_category_asr(self):
        """Test model category detection for ASR."""
        from src.cudara.quantization import ModelCategory, detect_model_category

        cat = detect_model_category(task="automatic-speech-recognition")
        assert cat == ModelCategory.ASR

    def test_detect_model_category_embedding(self):
        """Test model category detection for embeddings."""
        from src.cudara.quantization import ModelCategory, detect_model_category

        cat = detect_model_category(task="feature-extraction")
        assert cat == ModelCategory.EMBEDDING

    def test_quantization_profiles_exist(self):
        """Test that quantization profiles exist for all categories."""
        from src.cudara.quantization import QUANTIZATION_PROFILES, ModelCategory

        for category in ModelCategory:
            assert category in QUANTIZATION_PROFILES


class TestModels:
    """Tests for model configuration."""

    def test_model_config_parsing(self, mock_models_json):
        """Test parsing model configuration."""
        with open(mock_models_json) as f:
            models = json.load(f)

        assert "test-org/test-model" in models
        assert models["test-org/test-model"]["task"] == "text-generation"

    def test_model_config_quantization(self, mock_models_json):
        """Test quantization config parsing."""
        with open(mock_models_json) as f:
            models = json.load(f)

        model = models["test-org/test-model"]
        assert "quantization" in model
        assert not model["quantization"]["enabled"]

    def test_vlm_config_has_image_processing(self, mock_models_json):
        """Test VLM config has image processing settings."""
        with open(mock_models_json) as f:
            models = json.load(f)

        vlm = models["test-org/test-vlm"]
        assert "image_processing" in vlm
        assert vlm["image_processing"]["min_pixels"] > 0


class TestCLI:
    """Tests for CLI module."""

    def test_cli_import(self):
        """Test CLI module can be imported."""
        from src.cudara.cli import cmd_list, cmd_pull, main

        assert callable(main)
        assert callable(cmd_list)
        assert callable(cmd_pull)

    def test_get_config_default(self, temp_dir):
        """Test default config when no config file exists."""
        from src.cudara.cli import DEFAULT_HOST, get_config

        with patch.object(Path, "exists", return_value=False):
            config = get_config()
            assert config["host"] == DEFAULT_HOST


class TestAPIModels:
    """Tests for API data models."""

    def test_generate_request_model(self):
        """Test GenerateRequest model."""
        from src.cudara.main import GenerateRequest

        req = GenerateRequest(model="test-model", prompt="Hello", stream=False)
        assert req.model == "test-model"
        assert req.prompt == "Hello"
        assert req.options == {}

    def test_chat_message_model(self):
        """Test ChatMessage model."""
        from src.cudara.main import ChatMessage

        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.images is None

    def test_embedding_request_model(self):
        """Test EmbeddingRequest model."""
        from src.cudara.main import EmbeddingRequest

        req = EmbeddingRequest(model="test", input="Hello")
        assert req.model == "test"
        assert req.input == "Hello"

        req_list = EmbeddingRequest(model="test", input=["Hello", "World"])
        assert isinstance(req_list.input, list)


class TestErrorHandling:
    """Tests for error handling."""

    def test_app_error(self):
        """Test AppError exception."""
        from src.cudara.main import AppError, ErrorCode

        error = AppError("Test error", status_code=404, details={"code": ErrorCode.MODEL_NOT_FOUND})
        assert error.message == "Test error"
        assert error.status_code == 404
        assert error.details["code"] == ErrorCode.MODEL_NOT_FOUND

    def test_error_response_format(self):
        """Test error response format."""
        from src.cudara.main import error_response

        resp = error_response("test_error", "Test message", extra="data")
        assert resp["error"]["code"] == "test_error"
        assert resp["error"]["message"] == "Test message"
        assert resp["error"]["extra"] == "data"


class TestModelStatus:
    """Tests for model status enum."""

    def test_model_status_values(self):
        """Test ModelStatus enum values."""
        from src.cudara.main import ModelStatus

        assert ModelStatus.DOWNLOADING.value == "downloading"
        assert ModelStatus.QUANTIZING.value == "quantizing"
        assert ModelStatus.READY.value == "ready"
        assert ModelStatus.ERROR.value == "error"
