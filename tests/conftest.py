"""
Test Configuration
==================
Sets up the environment for tests, ensuring the 'src' directory is discoverable.
"""
import sys
import os
import json
from pathlib import Path
import pytest

# Add 'src' to sys.path to allow importing 'cudara' directly
# This fixes "ModuleNotFoundError: No module named 'src'"
ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

@pytest.fixture
def mock_models_json(tmp_path):
    """Create a temporary models.json for testing."""
    p = tmp_path / "models.json"
    content = {
        "test-org/test-model": {
            "task": "text-generation",
            "quantization": {"enabled": False}
        },
        "test-org/test-vlm": {
            "task": "image-to-text",
            "image_processing": {"min_pixels": 100}
        }
    }
    p.write_text(json.dumps(content))
    return p

@pytest.fixture
def temp_dir(tmp_path):
    """Alias for older tests expecting `temp_dir`."""
    return tmp_path