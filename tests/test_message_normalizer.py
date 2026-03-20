import pytest
from bio_ml_agent.core.message_normalizer import MessageNormalizer, StandardMessage

def test_string_input_normalization():
    msg = MessageNormalizer.normalize_input("Hello world", role="user")
    assert isinstance(msg, StandardMessage)
    assert msg.role == "user"
    assert msg.content == "Hello world"
    assert not msg.images
    assert not msg.files

def test_dict_input_normalization():
    # Gradio tarzı bir girdi
    raw = {"text": "Look at this image", "files": [{"path": "/tmp/img.png"}]}
    msg = MessageNormalizer.normalize_input(raw, role="user")
    
    assert msg.content == "Look at this image"
    assert msg.files == ["/tmp/img.png"]

def test_to_openai_format():
    msg = StandardMessage(role="user", content="Test", images=["b64_string"])
    payload = MessageNormalizer.to_provider_format(msg, provider="openai")
    
    assert payload["role"] == "user"
    assert isinstance(payload["content"], list)
    assert payload["content"][0]["type"] == "text"
    assert payload["content"][1]["type"] == "image_url"

def test_to_anthropic_format():
    msg = StandardMessage(role="user", content="Test", images=["data:image/jpeg;base64,12345"])
    payload = MessageNormalizer.to_provider_format(msg, provider="anthropic")
    
    assert payload["role"] == "user"
    assert payload["content"][1]["type"] == "image"
    assert payload["content"][1]["source"]["data"] == "12345"

def test_to_gemini_format():
    msg = StandardMessage(role="system", content="You are a bot")
    payload = MessageNormalizer.to_provider_format(msg, provider="gemini")
    
    assert payload["role"] == "user" # System map
    assert payload["parts"] == ["You are a bot"]
