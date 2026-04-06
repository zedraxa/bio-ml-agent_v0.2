
import os
import sys
from pathlib import Path
import logging

# Proje kökü importları
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from bio_ml_agent.ultra_agent.runtime.browser.browser_worker import BrowserWorker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

def test_worker_isolation(tmp_path):
    workspace = tmp_path / "bio_ml_agent_test_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    project = "test_p4_project"
    session1 = "session_alpha"
    session2 = "session_beta"
    
    worker1 = BrowserWorker(workspace=workspace, project_name=project, session_id=session1)
    worker2 = BrowserWorker(workspace=workspace, project_name=project, session_id=session2)
    
    print(f"--- Running Task 1 ({session1}) ---")
    # Done aksiyonu ile hemen bitmesini sağlayalım
    task1 = '{"thought": "Test 1", "action": {"type": "done", "value": "Success Alpha"}}'
    res1 = worker1.run_task(task1)
    print(f"Result 1: {res1}")
    
    print(f"\n--- Running Task 2 ({session2}) ---")
    task2 = '{"thought": "Test 2", "action": {"type": "done", "value": "Success Beta"}}'
    res2 = worker2.run_task(task2)
    print(f"Result 2: {res2}")
    
    # Artifact kontrolü
    base_path = workspace / "browser_artifacts" / project
    path1 = base_path / session1
    path2 = base_path / session2
    
    print(f"\nChecking artifacts...")
    print(f"Session 1 (Alpha) exists: {path1.exists()}")
    print(f"Session 1 Trace exists: {(path1 / 'trace.zip').exists()}")
    print(f"Session 2 (Beta) exists: {path2.exists()}")
    print(f"Session 2 Trace exists: {(path2 / 'trace.zip').exists()}")

if __name__ == "__main__":
    test_worker_isolation()
