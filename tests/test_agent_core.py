import pytest
from unittest.mock import MagicMock, patch
from core.agent_core import AgentCore
from core.config import AgentConfig
from exceptions import ToolExecutionError

@pytest.fixture
def mock_config():
    config = AgentConfig()
    config.model = "test-model"
    config.workspace = "/tmp/test_workspace"
    config.history_dir = "/tmp/test_history"
    return config

@pytest.fixture
def backend_mock():
    backend = MagicMock()
    # chat_stream returns a generator
    backend.chat_stream.return_value = ["Hello", " ", "World"]
    backend.chat.return_value = "Hello World"
    return backend

@patch("llm_backend.auto_create_backend")
def test_agent_core_initialization(mock_create_backend, mock_config, backend_mock):
    mock_create_backend.return_value = backend_mock
    
    core = AgentCore(mock_config)
    
    assert core.config == mock_config
    assert core.llm == backend_mock
    mock_create_backend.assert_called_once_with("test-model", mode="auto")

@patch("llm_backend.auto_create_backend")
def test_classify_intent(mock_create_backend, mock_config, backend_mock):
    mock_create_backend.return_value = backend_mock
    core = AgentCore(mock_config)
    
    # Test fallback classification 
    assert core.classify_intent("Merhaba nasılsın?") == "CHAT"
    assert core.classify_intent("Bir pandas DataFrame oluştur ve df.describe() çalıştır.") == "TOOL_LOOP"
    assert core.classify_intent("Swarm bana SVM eğitimi yapsın") == "SWARM"
    assert core.classify_intent("Buradaki meme kanseri veri setini bir ml_pipeline ile baştan sona incele.") == "ML_PIPELINE"

@patch("llm_backend.auto_create_backend")
def test_route_task_override(mock_create_backend, mock_config, backend_mock):
    """Test if intent override correctly bypasses classify_intent"""
    mock_create_backend.return_value = backend_mock
    core = AgentCore(mock_config)
    
    # Normally "Merhaba" is CHAT, but we'll override it to look like it routes to ML_PIPELINE
    with patch.object(core, "_tool_loop") as mock_tool_loop:
        mock_tool_loop.return_value = [{"type": "done"}]
        
        events = list(core.route_task("Merhaba", [], intent_override="ML_PIPELINE"))
        
        # event 1 is intent
        assert events[0] == {"type": "intent", "intent": "ML_PIPELINE"}
        # event 2+ is from mock_tool_loop
        assert events[1] == {"type": "done"}
        
        mock_tool_loop.assert_called_once()

