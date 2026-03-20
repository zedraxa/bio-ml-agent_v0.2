import pytest
from unittest.mock import patch, MagicMock
import json
from pathlib import Path

# We must import the module to patch its global variables
import bio_ml_agent.services.dashboard_service as ds

@pytest.fixture
def temp_workspace(tmp_path):
    # Setup some dummy workspace stuff
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    docs = tmp_path / "docs"
    docs.mkdir()
    
    (docs / "index.md").write_text("Test docs")
    (workspace / "RAPOR.md").write_text("# Test Rapor")
    
    return tmp_path

@patch("bio_ml_agent.services.dashboard_service.WORKSPACE_DIR")
@patch("bio_ml_agent.services.dashboard_service.BASE_DIR")
@patch("bio_ml_agent.services.dashboard_service.TASKS_FILE")
def test_task_lifecycle(mock_tasks_file, mock_base_dir, mock_workspace_dir, temp_workspace):
    # Route data to temp workspace
    mock_base_dir.return_value = temp_workspace
    mock_workspace_dir.return_value = temp_workspace / "workspace"
    
    dummy_tasks_file = temp_workspace / "tasks.json"
    ds.TASKS_FILE = dummy_tasks_file
    
    # Init empty
    if dummy_tasks_file.exists():
        dummy_tasks_file.unlink()
        
    ds.seed_tasks()
    assert dummy_tasks_file.exists()
    
    # Create
    task = ds.create_task("Test task", "Desc")
    assert task["title"] == "Test task"
    assert task["status"] == "pending"
    assert "id" in task
    
    task_id = task["id"]
    
    # Get all
    tasks = ds.list_tasks()
    # It will contain seed tasks + new task
    assert len(tasks) > 1
    
    # Update
    updated = ds.update_task(task_id, status="in_progress")
    assert updated["status"] == "in_progress"
    
    # Status endpoint (approve)
    approved = ds.approve_task(task_id)
    assert approved["status"] == "completed"
    
    # Reject
    rejected = ds.reject_task(task_id)
    assert rejected["status"] == "pending"
    
    # Delete
    assert ds.delete_task(task_id) is not None
    
    # Make sure it's gone
    for t in ds.list_tasks():
        assert t["id"] != task_id

@patch("bio_ml_agent.services.dashboard_service.REPORT_FILE")
def test_get_report(mock_report_file, temp_workspace):
    dummy_report = temp_workspace / "RAPOR.md"
    dummy_report.write_text("Hello Report", encoding="utf-8")
    ds.REPORT_FILE = dummy_report
    
    report = ds.get_report()
    assert "Hello Report" in report

