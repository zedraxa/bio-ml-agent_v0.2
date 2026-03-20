import pytest
import os
import shutil
from pathlib import Path
from bio_ml_agent.services.agent_service import AgentService

def test_checkpoint_and_recovery():
    workspace = Path("./tmp_test_workspace")
    if workspace.exists(): shutil.rmtree(workspace)
    workspace.mkdir()
    
    try:
        # 1. Start a session and message
        service = AgentService(model="mock-gpt", workspace=str(workspace))
        events = list(service.process_message("Create a test project"))
        
        proj_name = service.project_name
        sid = service.session_id
        checkpoint_path = service.project_root / "checkpoint.json"
        
        assert proj_name is not None
        assert checkpoint_path.exists()
        
        # 2. Create a NEW service instance and try to recover
        new_service = AgentService(model="mock-gpt", workspace=str(workspace))
        recovered = new_service.recover_last_session()
        
        assert recovered is True
        assert new_service.session_id == sid
        assert new_service.project_name == proj_name
        assert len(new_service.messages) > 1
        
    finally:
        if workspace.exists(): shutil.rmtree(workspace)

