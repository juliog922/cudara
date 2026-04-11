"""CLI command tests for Cudara."""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest
from httpx import Request, Response

from cudara.cli import (
    cmd_list,
    cmd_ps,
    cmd_rm,
    cmd_run,
    cmd_stop,
    extract_images_from_prompt,
    format_size,
)


@pytest.fixture
def mock_client():
    """Fixture providing a mocked httpx.Client via get_client context manager."""
    with patch("cudara.cli.get_client") as mock:
        client_instance = MagicMock()
        mock.return_value.__enter__ = MagicMock(return_value=client_instance)
        mock.return_value.__exit__ = MagicMock(return_value=False)
        yield client_instance


# ===================================================================
# Unit tests for helpers
# ===================================================================
class TestHelpers:
    """Tests for pure utility functions."""

    def test_format_size_zero(self):
        """Zero bytes formats correctly."""
        assert format_size(0) == "0 B"

    def test_format_size_bytes(self):
        """Small sizes stay in bytes."""
        assert format_size(512) == "512.0 B"

    def test_format_size_kb(self):
        """1024 bytes = 1.0 KB."""
        assert format_size(1024) == "1.0 KB"

    def test_format_size_gb(self):
        """~1 GB formats correctly."""
        assert "GB" in format_size(1024**3)

    def test_extract_images_no_images(self):
        """No image paths returns empty list."""
        text, images = extract_images_from_prompt("just a normal prompt")
        assert text == "just a normal prompt"
        assert images == []

    def test_extract_images_nonexistent_file(self):
        """Non-existent file path is kept as text."""
        text, images = extract_images_from_prompt("look at /nonexistent/photo.png please")
        assert "/nonexistent/photo.png" in text
        assert images == []


# ===================================================================
# Command tests
# ===================================================================
class TestListCommand:
    """Tests for the 'list' CLI command."""

    def test_list_models(self, mock_client, capsys):
        """List command prints model names and sizes."""
        req = Request("GET", "http://test")
        mock_client.get.return_value = Response(
            200,
            json={
                "models": [
                    {
                        "name": "Qwen-3B",
                        "size": 1024,
                        "digest": "sha256:1234567890abcdef",
                        "modified_at": "2026-01-01T00:00:00Z",
                    },
                ]
            },
            request=req,
        )

        cmd_list(argparse.Namespace(host="http://test"))
        captured = capsys.readouterr()
        assert "Qwen-3B" in captured.out
        assert "1.0 KB" in captured.out

    def test_list_empty(self, mock_client, capsys):
        """List with no models prints message."""
        req = Request("GET", "http://test")
        mock_client.get.return_value = Response(200, json={"models": []}, request=req)

        cmd_list(argparse.Namespace(host="http://test"))
        captured = capsys.readouterr()
        assert "No models" in captured.out


class TestPsCommand:
    """Tests for the 'ps' CLI command."""

    def test_ps_empty(self, mock_client, capsys):
        """No running models prints message."""
        req = Request("GET", "http://test")
        mock_client.get.return_value = Response(200, json={"models": []}, request=req)

        cmd_ps(argparse.Namespace(host="http://test"))
        captured = capsys.readouterr()
        assert "No models" in captured.out

    def test_ps_with_model(self, mock_client, capsys):
        """Running model is displayed."""
        req = Request("GET", "http://test")
        mock_client.get.return_value = Response(
            200,
            json={
                "models": [
                    {
                        "name": "Qwen-3B",
                        "digest": "sha256:abcdef123456",
                        "size_vram": 2 * 1024**3,
                        "expires_at": "2026-01-01T01:00:00Z",
                    },
                ]
            },
            request=req,
        )

        cmd_ps(argparse.Namespace(host="http://test"))
        captured = capsys.readouterr()
        assert "Qwen-3B" in captured.out


class TestRmCommand:
    """Tests for the 'rm' CLI command."""

    def test_rm_success(self, mock_client, capsys):
        """Successful delete prints confirmation."""
        req = Request("DELETE", "http://test")
        mock_client.request.return_value = Response(200, request=req)

        cmd_rm(argparse.Namespace(host="http://test", model="Qwen-3B"))
        captured = capsys.readouterr()
        assert "deleted 'Qwen-3B'" in captured.out


class TestStopCommand:
    """Tests for the 'stop' CLI command."""

    def test_stop_sends_keep_alive_zero(self, mock_client, capsys):
        """Stop command sends keep_alive=0."""
        req = Request("POST", "http://test")
        mock_client.post.return_value = Response(200, request=req)

        cmd_stop(argparse.Namespace(host="http://test", model="Qwen-3B"))
        mock_client.post.assert_called_with("/api/generate", json={"model": "Qwen-3B", "keep_alive": 0})


class TestRunCommand:
    """Tests for the 'run' CLI command."""

    def test_single_prompt(self, mock_client, capsys):
        """Single prompt mode streams and prints output."""
        # Mock show (model exists)
        req = Request("POST", "http://test")
        mock_client.post.return_value = Response(200, json={}, request=req)

        # Mock streaming response
        mock_stream = MagicMock()
        mock_stream.status_code = 200
        mock_stream.raise_for_status = MagicMock()
        mock_stream.iter_lines.return_value = [
            '{"response": "Hello ", "done": false}',
            '{"response": "World!", "done": true}',
        ]
        mock_client.stream.return_value.__enter__ = MagicMock(return_value=mock_stream)
        mock_client.stream.return_value.__exit__ = MagicMock(return_value=False)

        cmd_run(argparse.Namespace(host="http://test", model="Qwen-3B", prompt=["Say", "hello"]))
        captured = capsys.readouterr()
        assert "Hello " in captured.out
        assert "World!" in captured.out
