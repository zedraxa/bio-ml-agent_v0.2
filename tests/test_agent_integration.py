# tests/test_agent_integration.py
import os
import shutil
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from bio_ml_agent.services.agent_service import AgentService

@pytest.fixture
def temp_workspace(tmp_path):
    """Temporary workspace for testing."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace

@pytest.fixture
def agent_service(temp_workspace):
    """Initializes AgentService with a temporary workspace."""
    # Mock config to point to temp workspace
    with patch("bio_ml_agent.services.agent_service.load_config") as mock_cfg:
        mock_cfg.return_value = MagicMock()
        mock_cfg.return_value.agent.model = "gpt-4o"
        mock_cfg.return_value.workspace.base_dir = str(temp_workspace)
        mock_cfg.return_value.agent.timeout = 300
        mock_cfg.return_value.agent.max_steps = 50
        mock_cfg.return_value.history.directory = str(temp_workspace / "history")
        (temp_workspace / "history").mkdir()
        
        service = AgentService(model="gpt-4o", workspace=str(temp_workspace))
        return service

def test_project_lifecycle_initialization(agent_service, temp_workspace):
    """Tests that AgentService creates a project context on first message."""
    user_msg = "Create a new project for diabetes analysis"
    
    # We mock AgentCore.route_task to avoid LLM calls
    with patch("bio_ml_agent.core.agent_core.AgentCore.route_task") as mock_route:
        mock_route.return_value = iter([{"type": "status", "content": "Starting"}])
        
        # Trigger process_message
        list(agent_service.process_message(user_msg))
        
        # Verify project context
        assert agent_service.project_name is not None
        assert "create-a-new-project-for-diabetes" in agent_service.project_name
        assert agent_service.project_root.exists()
        assert (agent_service.project_root / "project.json").exists()

def test_session_save_load(agent_service, temp_workspace):
    """Tests session persistence across AgentService instances."""
    user_msg = "Initial message"
    session_id = agent_service.session_id
    history_dir = Path(agent_service.config.history_dir)
    
    # 1. Simulate workflow and save (AgentService currently saves via core/conversation tools if called)
    # For this test, we manually save and then load via set_session
    messages = [{"role": "user", "content": user_msg}]
    metadata = {"project_name": "test_proj", "project_path": str(temp_workspace / "test_proj")}
    (temp_workspace / "test_proj").mkdir()
    
    from bio_ml_agent.core.conversation import save_conversation
    save_conversation(history_dir, session_id, messages, metadata)
    
    # 2. Create new instance and load
    new_service = AgentService(workspace=str(temp_workspace))
    from bio_ml_agent.core.conversation import load_conversation
    loaded_msgs, loaded_meta = load_conversation(history_dir, session_id)
    new_service.set_session(session_id, loaded_msgs, loaded_meta)
    
    assert new_service.session_id == session_id
    assert len(new_service.messages) == 1
    assert new_service.project_name == "test_proj"

def test_orchestration_flow_with_tool_output(agent_service):
    """Tests that process_message correctly routes events and tool outputs."""
    user_msg = "Run some analysis"
    
    with patch("bio_ml_agent.core.agent_core.AgentCore.route_task") as mock_route:
        # Mocking a sequence of events
        mock_route.return_value = iter([
            {"type": "status", "content": "Tool call started"},
            {"type": "tool_output", "tool": "PYTHON", "output": "Result: 42"},
            {"type": "assistant", "content": "The answer is 42."}
        ])
        
        events = list(agent_service.process_message(user_msg))
        
        types = [e["type"] for e in events]
        assert "status" in types
        assert "tool_output" in types
        assert "assistant" in types
        
def test_mock_tool_execution(temp_workspace):
    """Tests that AgentService correctly handles tool calls using MockBackend."""
    # Create service with mock model
    service = AgentService(model="mock-gpt", workspace=str(temp_workspace))
    
    # We want a response that triggers a Python tool call
    mock_resp = "<PYTHON>print('Test 123')</PYTHON>\nDone."
    
    # Now trigger a real process_message call
    with patch("bio_ml_agent.llm_backend.auto_create_backend") as mock_backend_factory:
        from bio_ml_agent.llm_backend import MockBackend
        mock_llm = MockBackend(responses=[mock_resp, "Final response"])
        mock_backend_factory.return_value = mock_llm
        
        events = list(service.process_message("Execute test code"))
        
        types = [e["type"] for e in events]
        assert "tool_start" in types
        assert "tool_output" in types
        assert "assistant" in types
        
        # Verify Python was called
        python_ev = next(e for e in events if e.get("tool") == "PYTHON")
        assert python_ev is not None
