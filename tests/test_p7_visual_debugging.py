"""
Phase 16 (P7) — İnsan Gözüne Görünür Debugging
BrowserSubAgent execute loop'unda klasörlerin ve dosyaların doğru yaratıldığını test eder.
"""

import sys
import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, call

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

PASS = 0
FAIL = 0

def ok(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}", flush=True)
    else:
        FAIL += 1
        print(f"  ❌ {name} — {detail}", flush=True)

def test_visual_debugging_artifacts():
    print("\n=== Test 1: P7 Visual Debugging Artifact Generation ===", flush=True)
    import tempfile
    
    # Fake LLM Backend dictating a click action
    fake_llm_response = json.dumps({
        "thought": "I will click the button.",
        "action": {"type": "click", "target": {"bio_id": "bio-1"}}
    })

    # Prepare Mocks
    mock_backend = MagicMock()
    mock_backend.chat.return_value = fake_llm_response

    # Import and patch inner modules
    from unittest.mock import patch
    import sys
    
    # Fake module for llm_backend to be importable inside execute()
    mock_llm_module = MagicMock()
    mock_llm_module.auto_create_backend.return_value = mock_backend
    sys.modules['llm_backend'] = mock_llm_module
    
    with patch("ultra_agent.runtime.browser.browser_agent.AuditTrailLogger"):
            with tempfile.TemporaryDirectory() as td:
                mock_page = MagicMock()
                # Simulate valid locator for P6 validation
                mock_loc = MagicMock()
                mock_loc.count.return_value = 1
                mock_loc.is_visible.return_value = True
                mock_loc.is_disabled.return_value = False
                mock_loc.first = mock_loc  # Fix nested .first.first calls
                mock_page.locator.return_value = mock_loc
                mock_page.get_by_role = lambda *a, **k: mock_loc
                mock_page.get_by_text = lambda *a, **k: mock_loc
                
                # Setup job dir as BrowserWorker would
                job_dir = Path(td) / "job_123"
                job_dir.mkdir()
                mock_page._job_dir = job_dir

                from ultra_agent.runtime.browser.browser_agent import BrowserSubAgent
                agent = BrowserSubAgent(workspace=Path(td), project_name="p", session_id="s", max_steps=1)
                
                # Mock distiller output
                mock_page.evaluate.return_value = {
                    "page": {"title": "Test", "url": "http://test"},
                    "interactive": [{"bio_id": "bio-1", "pw_role": "button", "pw_name": "Submit"}]
                }

                # Run execute for 1 step
                agent.execute("Click the button", mock_page)

                # Verify Directory Structure
                actions_log = job_dir / "actions.jsonl"
                ok("actions.jsonl created", actions_log.exists())
                
                step_dir = job_dir / "steps" / "01_click"
                ok("step directory renamed to include action type", step_dir.exists())
                
                if step_dir.exists():
                    ok("dom_snapshot.html exists", (step_dir / "dom_snapshot.html").exists())
                    ok("step_meta.json exists", (step_dir / "step_meta.json").exists())
                    
                    # Verify page.screenshot was called for before, candidates, after
                    screenshots = [c_args[1].get('path', '') or (c_args[0][0] if c_args[0] else '') 
                                   for c_args in mock_page.screenshot.call_args_list]
                    
                    before_ok = any("before.png" in str(s) for s in screenshots)
                    cands_ok = any("candidates.png" in str(s) for s in screenshots)
                    after_ok = any("after.png" in str(s) for s in screenshots)
                    
                    ok("before.png captured", before_ok)
                    ok("candidates.png captured", cands_ok)
                    ok("after.png captured", after_ok)
                    
                    # Check step_meta content
                    meta = json.loads((step_dir / "step_meta.json").read_text())
                    ok("step_meta contains thought", meta["thought"] == "I will click the button.")
                    ok("step_meta contains action type", meta["action"]["type"] == "click")
                    
                    # Check overlay cleanup
                    evaluations = [c_args[0][0] for c_args in mock_page.evaluate.call_args_list]
                    cleanup_ok = any("bio-ml-overlay" in str(e) and "remove" in str(e) for e in evaluations)
                    ok("overlay cleanup evaluated", cleanup_ok)

if __name__ == "__main__":
    print("=" * 60)
    print("  Phase 16 (P7) — Visual Debugging Verification")
    print("=" * 60)
    
    test_visual_debugging_artifacts()

    print(f"\n{'=' * 60}")
    print(f"  Sonuç: {PASS} geçti, {FAIL} başarısız")
    print(f"{'=' * 60}")
    sys.exit(0 if FAIL == 0 else 1)
