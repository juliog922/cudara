"""
Test Configuration Module.
==========================
Sets up the testing environment, specifically ensuring that the 'src' directory
is discoverable by the Python interpreter during test execution. This allows
direct imports of the 'cudara' package.
"""

import json
import sys
from pathlib import Path

import pytest

# Add 'src' to sys.path to allow importing 'cudara' directly.
# This logic ensures that regardless of where pytest is invoked from,
# the internal package structure remains valid.
ROOT_DIR: Path = Path(__file__).parent.parent
SRC_DIR: Path = ROOT_DIR / "src"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture
def mock_models_json(tmp_path: Path) -> Path:
    """
    Create a temporary 'models.json' file populated with dummy data for testing.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.

    Returns:
        Path: The file path to the created temporary models.json.
    """
    p: Path = tmp_path / "models.json"
    content: dict[str, dict] = {
        "test-org/test-model": {
            "task": "text-generation",
            "quantization": {"load_in_4bit": False},
        },
        "test-org/test-vlm": {
            "task": "image-to-text",
            "image_processing": {"min_pixels": 100},
        },
    }
    p.write_text(json.dumps(content))
    return p


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """
    Provide a temporary directory path for tests.
    Acts as an alias for the standard 'tmp_path' fixture for compatibility.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.

    Returns:
        Path: The temporary directory path.
    """
    return tmp_path
