import argparse
from unittest.mock import MagicMock, patch

import pytest
from httpx import Request, Response

from cudara.cli import cmd_list, cmd_rm, cmd_run, cmd_stop


@pytest.fixture
def mock_client():
    """Fixture to mock the HTTP client."""
    with patch("cudara.cli.get_client") as mock:
        client_instance = MagicMock()
        mock.return_value.__enter__.return_value = client_instance
        yield client_instance


def test_cli_list(mock_client, capsys):
    """Test the list command output."""
    # Attach a dummy request so raise_for_status() doesn't fail
    req = Request("GET", "http://test")
    mock_client.get.return_value = Response(
        200, json={"models": [{"name": "Qwen-3B", "size": 1024, "digest": "sha256:1234567890abcdef"}]}, request=req
    )

    args = argparse.Namespace(host="http://test")
    cmd_list(args)

    captured = capsys.readouterr()
    assert "Qwen-3B" in captured.out
    assert "1.0 KB" in captured.out


def test_cli_rm(mock_client, capsys):
    """Test delete command output."""
    req = Request("DELETE", "http://test")
    mock_client.request.return_value = Response(200, request=req)

    args = argparse.Namespace(host="http://test", model="Qwen-3B")
    cmd_rm(args)

    captured = capsys.readouterr()
    assert "deleted 'Qwen-3B'" in captured.out


def test_cli_stop(mock_client, capsys):
    """Test stop command output."""
    req = Request("POST", "http://test")
    mock_client.post.return_value = Response(200, request=req)

    args = argparse.Namespace(host="http://test", model="Qwen-3B")
    cmd_stop(args)

    mock_client.post.assert_called_with("/api/generate", json={"model": "Qwen-3B", "keep_alive": 0})


def test_cli_run_single_prompt(mock_client, capsys, monkeypatch):
    """Test run command output with single prompt."""
    req = Request("POST", "http://test")
    mock_client.post.return_value = Response(200, request=req)

    mock_stream_response = MagicMock()
    mock_stream_response.iter_lines.return_value = [
        '{"response": "Hello ", "done": false}',
        '{"response": "World!", "done": true}',
    ]
    mock_client.stream.return_value.__enter__.return_value = mock_stream_response

    args = argparse.Namespace(host="http://test", model="Qwen-3B", prompt=["Say", "hello"])
    cmd_run(args)

    captured = capsys.readouterr()
    assert "Hello World!" in captured.out
