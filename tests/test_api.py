import base64


def test_api_tags(client):
    """Test retrieving available models."""
    response = client.get("/api/tags")
    assert response.status_code == 200
    assert "models" in response.json()
    assert len(response.json()["models"]) == 5


def test_generate_non_streaming(client):
    """Test generate endpoint with non streaming option."""
    payload = {"model": "Qwen/Qwen2.5-3B-Instruct-AWQ", "prompt": "Write a haiku", "stream": False}
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "Qwen/Qwen2.5-3B-Instruct-AWQ"
    assert "response" in data
    assert data["done"] is True


def test_generate_structured_json(client):
    """Test generate endpoint forcing json format."""
    payload = {
        "model": "Qwen/Qwen2.5-3B-Instruct-AWQ",
        "prompt": "List 3 colors in a JSON array.",
        "format": "json",
        "stream": False,
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200
    assert response.json()["response"] == "Mocked JSON"


def test_generate_audio_transcription(client):
    """Test generate endpoint with audio file."""
    # Dummy base64 encoded audio
    dummy_audio = base64.b64encode(b"ID3...dummydata").decode()
    payload = {
        "model": "openai/whisper-base",
        "prompt": "Transcribe this",
        "images": [dummy_audio],
        "options": {"is_audio": True},
        "stream": False,
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200
    assert response.json()["response"] == "Mocked audio transcription."


def test_chat_streaming(client):
    """Test chat endpoint with streaming option."""
    payload = {
        "model": "Qwen/Qwen2.5-3B-Instruct-AWQ",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }
    with client.stream("POST", "/api/chat", json=payload) as response:
        assert response.status_code == 200
        # Read the first NDJSON line
        lines = list(response.iter_lines())
        assert len(lines) > 0
        data = __import__("json").loads(lines[0])
        assert "message" in data


def test_embed_rerank(client):
    """Test embed endpoint with rerank option."""
    payload = {"model": "BAAI/bge-reranker-v2-m3", "input": ["Query", "Doc 1", "Doc 2"], "options": {"is_rerank": True}}
    response = client.post("/api/embed", json=payload)
    assert response.status_code == 200
    assert "scores" in response.json()
    assert len(response.json()["scores"]) == 2


def test_model_not_found(client):
    """Test generate endpoint with unexisted model."""
    payload = {"model": "fake-model-123", "prompt": "hello", "stream": False}
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 404
    assert "not found" in response.json()["error"]


def test_malformed_json_validation(client):
    """Test chat endpoint with malformed json."""
    # Send empty payload violating Pydantic schema
    response = client.post("/api/chat", json={"model": "Qwen/Qwen2.5-3B-Instruct-AWQ"})
    assert response.status_code == 400
    assert "Invalid request format" in response.json()["error"]
