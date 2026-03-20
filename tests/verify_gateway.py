import sys
from pathlib import Path

# Add src to sys.path
src_path = str(Path(__file__).resolve().parent.parent / "src")
sys.path.append(src_path)

from bio_ml_agent.core.agent_core import AgentCore
from bio_ml_agent.core.config import AgentConfig as CoreAgentConfig
from bio_ml_agent.utils.config import AppConfig, GatewayConfig

def test_remote_mode_security():
    print("Testing Remote Mode Security...")
    
    # Setup config with remote_mode enabled
    app_config = AppConfig()
    app_config.gateway = GatewayConfig(enabled=True, remote_mode=True)
    
    # Mock workspace for AgentCore
    workspace_path = Path("/tmp/test_workspace_remote")
    workspace_path.mkdir(exist_ok=True)
    
    # Initialize AgentCore
    core_config = CoreAgentConfig(workspace=workspace_path)
    agent = AgentCore(config=core_config)
    
    # Inject gateway config manually (since core_config is different from app_config)
    # In real use, AgentCore would have a 'gateway' attribute in its config if updated
    agent.config.gateway = app_config.gateway
    
    # Test BASH execution
    result = agent._execute_tool("BASH", {"payload": "ls -la"})
    print(f"BASH Result: {result}")
    assert "restricted in Remote Mode" in result
    
    # Test WRITE_FILE execution
    result = agent._execute_tool("WRITE_FILE", {"payload": "test content", "attrs": {"path": "test.txt"}})
    print(f"WRITE_FILE Result: {result}")
    assert "restricted in Remote Mode" in result
    
    # Test allowed tool (e.g. LLM_QUERY if it existed or a dummy)
    # Let's say we check a non-restricted one
    # Note: _execute_tool fallback to actual execution if not restricted
    # We just want to see it doesn't get blocked by the new guard
    
    print("Remote Mode Security Test Passed!")

if __name__ == "__main__":
    try:
        test_remote_mode_security()
    except Exception as e:
        print(f"Test Failed: {e}")
        sys.exit(1)
