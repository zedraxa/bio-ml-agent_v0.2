import pytest
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agent import (
    write_file,
    run_bash,
    save_conversation,
    load_conversation,
    AgentConfig
)

def test_full_project_creation_mock(tmp_path):
    """Test full project creation logic through mocked sequence of tools"""
    os.environ["AGENT_PROJECT"] = "e2e_mock_proj"
    workspace = tmp_path
    
    # 1. Write training script
    payload1 = "path: e2e_mock_proj/src/train.py\n---\nprint('Training model')"
    res1 = write_file(payload1, workspace)
    assert "[OK]" in res1
    
    # 2. Add requirements
    payload2 = "path: e2e_mock_proj/requirements.txt\n---\npandas\nnumpy"
    res2 = write_file(payload2, workspace)
    assert "[OK]" in res2
    
    # Verify files created
    assert (workspace / "e2e_mock_proj" / "src" / "train.py").exists()
    assert (workspace / "e2e_mock_proj" / "requirements.txt").exists()


def test_write_file_path_integrity(tmp_path):
    """Test write_file strips workspace prefixes and maintains integrity"""
    os.environ["AGENT_PROJECT"] = "test_path_proj"
    workspace = tmp_path
    
    # Simulate LLM returning a bad path starting with workspace/
    payload = "path: workspace/test_path_proj/data/raw/data.csv\n---\na,b,c\n1,2,3"
    result = write_file(payload, workspace)
    
    assert "[OK]" in result
    
    # The file should be at workspace / 'test_path_proj' / 'data' / 'raw' / 'data.csv'
    expected_path = workspace / "test_path_proj" / "data" / "raw" / "data.csv"
    assert expected_path.exists()
    assert "1,2,3" in expected_path.read_text()


def test_bash_cwd_correctness(tmp_path):
    """Test that BASH commands execute inside the project directory correctly"""
    os.environ["AGENT_PROJECT"] = "test_bash_cwd"
    workspace = tmp_path
    
    # Create the dir first
    proj_dir = workspace / "test_bash_cwd"
    proj_dir.mkdir(parents=True, exist_ok=True)
    
    # run bash to print pwd into a file using > pwd.txt
    res = run_bash("pwd", proj_dir)
    assert str(proj_dir) in res


def test_conversation_save_load(tmp_path):
    """Test E2E conversation save and load loop"""
    history_dir = tmp_path / "sessions"
    history_dir.mkdir(parents=True)
    
    session_id = "e2e_session_123"
    messages = [
        {"role": "system", "content": "You are a helpful AI engineer."},
        {"role": "user", "content": "Please create a machine learning pipeline."},
        {"role": "assistant", "content": "I will do that right away."}
    ]
    meta = {"test": True, "created_at": "now"}
    
    # Save it
    save_conversation(history_dir, session_id, messages, meta)
    
    # Verify file
    session_file = history_dir / f"{session_id}.json"
    assert session_file.exists()
    
    # Load it
    loaded_messages, loaded_meta = load_conversation(history_dir, session_id)
    assert len(loaded_messages) == 3
    assert loaded_messages[2]["content"] == "I will do that right away."
    assert "created_at" in loaded_meta
