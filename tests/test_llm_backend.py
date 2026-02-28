import pytest
from unittest.mock import patch, MagicMock
from exceptions import LLMConnectionError
from llm_backend import (
    GeminiBackend, OllamaBackend, OpenAIBackend, AnthropicBackend,
    auto_create_backend, create_backend, detect_backend_name
)

def test_gemini_backend_init_missing_key(monkeypatch):
    """Test GeminiBackend initialization when API key is missing."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    backend = GeminiBackend(model="gemini-2.5-flash", api_key=None)
    # is_available should return False when no key is present
    assert backend.is_available() == False
    
    # chat should raise LLMConnectionError
    with pytest.raises(LLMConnectionError) as excinfo:
        backend.chat([{"role": "user", "content": "Hello"}])
    assert "GEMINI_API_KEY" in str(excinfo.value)

@patch("ollama.Client")
def test_ollama_backend_chat_mock(mock_client_class):
    """Test OllamaBackend chat method with a mocked ollama client."""
    mock_client = MagicMock()
    mock_client.chat.return_value = {"message": {"content": "Mocked Ollama response"}}
    mock_client_class.return_value = mock_client
    
    backend = OllamaBackend(model="test-model")
    response = backend.chat([{"role": "user", "content": "Hi"}])
    
    assert response == "Mocked Ollama response"
    mock_client.chat.assert_called_once_with(
        model="test-model",
        messages=[{"role": "user", "content": "Hi"}]
    )

def test_auto_backend_selection():
    """Test auto_create_backend to ensure it selects the right backend via model name."""
    # Test OpenAI detection
    backend = auto_create_backend("gpt-4o")
    assert isinstance(backend, OpenAIBackend)
    assert backend.model == "gpt-4o"

    # Test Anthropic detection
    backend = auto_create_backend("claude-3-opus-20240229")
    assert isinstance(backend, AnthropicBackend)
    
    # Test Gemini detection
    backend = auto_create_backend("gemini-2.5-flash")
    assert isinstance(backend, GeminiBackend)
    
    # Test fallback to Ollama
    backend = auto_create_backend("qwen2.5:7b")
    assert isinstance(backend, OllamaBackend)

@patch.dict("sys.modules", {"openai": MagicMock()})
def test_connection_error_handling(monkeypatch):
    """Test OpenAIBackend raises LLMConnectionError upon API error."""
    import sys
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    
    # Setup mock to raise an Exception during chat creation
    mock_openai = sys.modules["openai"]
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("API Timeout")
    mock_openai.OpenAI.return_value = mock_client
    
    backend = OpenAIBackend(model="gpt-4")
    
    with pytest.raises(LLMConnectionError) as excinfo:
        backend.chat([{"role": "user", "content": "Hello"}])
    
    assert "API Timeout" in str(excinfo.value)
