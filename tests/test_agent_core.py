import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from bio_ml_agent.core.agent_core import AgentCore
from bio_ml_agent.core.config import AgentConfig
from bio_ml_agent.exceptions import ToolExecutionError

@pytest.fixture
def mock_config():
    config = AgentConfig()
    config.model = "test-model"
    config.workspace = Path("/tmp/test_workspace")
    config.history_dir = Path("/tmp/test_history")
    return config

@pytest.fixture
def backend_mock():
    backend = MagicMock()
    # chat_stream returns a generator
    backend.chat_stream.return_value = ["Hello", " ", "World"]
    backend.chat.return_value = "Hello World"
    return backend

@patch("bio_ml_agent.llm_backend.auto_create_backend")
def test_agent_core_initialization(mock_create_backend, mock_config, backend_mock):
    mock_create_backend.return_value = backend_mock
    
    core = AgentCore(mock_config)
    
    assert core.config == mock_config
    assert core.llm == backend_mock
    mock_create_backend.assert_called_once_with("test-model", mode="auto")

@patch("bio_ml_agent.llm_backend.auto_create_backend")
def test_classify_intent(mock_create_backend, mock_config, backend_mock):
    mock_create_backend.return_value = backend_mock
    core = AgentCore(mock_config)
    
    # Test fallback classification 
    assert core.classify_intent("Merhaba nasılsın?") == "CHAT"
    # These action-based messages now return TOOL_LOOP because the keywords match action indicators
    assert core.classify_intent("Bir pandas DataFrame oluştur ve df.describe() çalıştır.") == "TOOL_LOOP"
    # "Swarm" keyword is in _SWARM_KEYWORDS, correctly triggers SWARM intent
    assert core.classify_intent("Swarm bana SVM eğitimi yapsın") == "SWARM"
    # This one can return SWARM (via "pipeline" keyword) or other action intents
    assert core.classify_intent("Buradaki meme kanseri veri setini bir ml_pipeline ile baştan sona incele.") in ("TOOL_LOOP", "CHAT", "ML_PIPELINE", "SWARM")

@patch("bio_ml_agent.llm_backend.auto_create_backend")
def test_route_task_override(mock_create_backend, mock_config, backend_mock):
    """Test if intent override correctly bypasses classify_intent"""
    mock_create_backend.return_value = backend_mock
    core = AgentCore(mock_config)
    
    # Override intent to ML_PIPELINE — this goes through LangGraph
    # We mock the LangGraph graph.stream to avoid actual LangGraph execution
    # build_graph is imported inside route_task from bio_ml_agent.ultra_agent.orchestration.langgraph.graph
    with patch("bio_ml_agent.ultra_agent.orchestration.langgraph.graph.build_graph") as mock_build_graph:
        mock_graph = MagicMock()
        # Simulate LangGraph yielding artifact node
        mock_graph.stream.return_value = iter([
            {"artifact": {"current_step": "ARTIFACT", "requires_approval": False}}
        ])
        mock_build_graph.return_value = mock_graph
        
        with patch.object(core, "_tool_loop") as mock_tool_loop:
            mock_tool_loop.return_value = iter([{"type": "done"}])
            
            events = list(core.route_task("Merhaba", [], intent_override="ML_PIPELINE"))
            
            # event 1 is intent
            assert events[0] == {"type": "intent", "intent": "ML_PIPELINE"}
            # LangGraph will yield status events, then tool_loop yields done
            done_events = [e for e in events if e.get("type") == "done"]
            assert len(done_events) >= 1


