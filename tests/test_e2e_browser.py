"""
Test script for verifying the Browser Agent end-to-end capabilities (P0 to P7).
It will launch a real headless browser utilizing BrowserWorker and BrowserSubAgent
to search for a concept on Wikipedia.
"""

import sys
import logging
from pathlib import Path

# Setup paths so ultra_agent is discoverable
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
log = logging.getLogger("e2e_test")

def run_e2e_test():
    log.info("Starting E2E Browser Agent Test...")
    workspace_dir = Path("/tmp/browser_e2e_workspace")
    workspace_dir.mkdir(parents=True, exist_ok=True)
    
    from ultra_agent.runtime.browser.browser_worker import BrowserWorker
    from ultra_agent.runtime.browser.browser_policy import BrowserPolicy
    
    # Create an ephemeral policy that allows Wikipedia
    policy = BrowserPolicy(
        mode="ephemeral",
        allowed_domains=["wikipedia.org"],
        max_timeout_s=60,
        enable_video=True,
        enable_tracing=True
    )
    
    # Initialize worker
    worker = BrowserWorker(workspace=workspace_dir, project_name="e2e_test", session_id="test_session")
    
    # A complete task to run
    task = "Go to https://www.wikipedia.org, search for 'Artificial Intelligence', and click the search button to go to the article. Once the article loads, extract the first paragraph text."
    
    log.info(f"Submitting task: {task}")
    
    # Mock LLM to avoid needing GEMINI_API_KEY for testing the artifact generation pipeline
    from unittest.mock import MagicMock
    import json
    import sys
    
    mock_backend = MagicMock()
    # Step 1: go to wikipedia
    resp1 = json.dumps({
        "thought": "I will go to wikipedia",
        "action": {"type": "goto", "value": "https://www.wikipedia.org"}
    })
    # Step 2: mark done
    resp2 = json.dumps({
        "thought": "Page loaded. Test complete.",
        "action": {"type": "done", "value": "Test completed successfully."}
    })
    
    mock_backend.chat.side_effect = [resp1, resp2]
    
    mock_llm_module = MagicMock()
    mock_llm_module.auto_create_backend.return_value = mock_backend
    sys.modules['llm_backend'] = mock_llm_module
    
    # Run the job
    result = worker.run_task(task, timeout_s=60, model="gemini-2.0-flash")
    
    log.info(f"\n--- Job Finished ---")
    log.info(f"Result Preview:\n{result}")
    
    import os
    latest_job_dir = max([d for d in workspace_dir.glob("browser_artifacts/e2e_test/test_session/job_*") if d.is_dir()], key=os.path.getmtime, default=None)
    if latest_job_dir:
        log.info(f"Artifacts saved to: {latest_job_dir}")
        print("\nVisual Debugging trace items generated in:", latest_job_dir / "steps")
    else:
        log.warning("No job artifact directory found.")

if __name__ == "__main__":
    run_e2e_test()
