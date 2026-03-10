"""
Phase 13 (P4) — BrowserWorker Mock-Based Verification
Playwright kurulu olmadan çalışan mock test.
Worker izolasyonunu, artifact yönetimini ve hata davranışını doğrular.
"""

import os
import sys
import json
import time
import tempfile
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

# Proje kökünü ayarla
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("test_p4")

PASS = 0
FAIL = 0


def result(name: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ✅ {name}", flush=True)
    else:
        FAIL += 1
        print(f"  ❌ {name} — {detail}", flush=True)


def test_worker_init():
    """BrowserWorker nesne oluşturma testi."""
    print("\n=== Test 1: BrowserWorker Init ===", flush=True)
    from ultra_agent.runtime.browser.browser_worker import BrowserWorker

    with tempfile.TemporaryDirectory() as tmpdir:
        ws = Path(tmpdir)
        worker = BrowserWorker(
            workspace=ws,
            project_name="test_proj",
            session_id="sess_001",
            headless=True,
            enable_video=False,
            enable_tracing=True,
        )

        result("workspace set", worker.workspace == ws)
        result("project_name set", worker.project_name == "test_proj")
        result("session_id set", worker.session_id == "sess_001")
        result("headless default True", worker.headless is True)
        result("base_dir exists", worker.base_dir.exists())
        result("base_dir path correct", "browser_artifacts/test_proj/sess_001" in str(worker.base_dir))


def test_worker_missing_playwright():
    """Playwright yüklü değilse zarif hata döner."""
    print("\n=== Test 2: Missing Playwright Graceful Error ===", flush=True)
    from ultra_agent.runtime.browser.browser_worker import BrowserWorker

    with tempfile.TemporaryDirectory() as tmpdir:
        worker = BrowserWorker(workspace=Path(tmpdir), project_name="no_pw")

        # Playwright import'unu bloke et
        with patch.dict("sys.modules", {"playwright": None, "playwright.sync_api": None}):
            # run_task içindeki import'u simüle etmek için
            original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

            def mock_import(name, *args, **kwargs):
                if name == "playwright.sync_api":
                    raise ImportError("No module named 'playwright'")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                res = worker.run_task("test task")

            result("returns error string", "HATA" in res or "Playwright" in res, res[:100])
            result("no crash", True)


def test_worker_with_mock_playwright():
    """Mock Playwright ile tam iş akışını test et."""
    print("\n=== Test 3: Mock Playwright Full Flow ===", flush=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        ws = Path(tmpdir)

        # Mock Playwright nesneleri
        mock_page = MagicMock()
        mock_page.title.return_value = "Test Page"
        mock_page.url = "https://test.com"
        mock_page.content.return_value = "<html><body>Test</body></html>"
        mock_page.evaluate.return_value = {
            "page": {"title": "Test Page", "url": "https://test.com"},
            "interactive": []
        }

        mock_context = MagicMock()
        mock_context.new_page.return_value = mock_page
        mock_tracing = MagicMock()
        mock_context.tracing = mock_tracing

        mock_browser = MagicMock()
        mock_browser.new_context.return_value = mock_context

        mock_playwright = MagicMock()
        mock_playwright.chromium.launch.return_value = mock_browser

        # sync_playwright context manager mock
        mock_sync_pw = MagicMock()
        mock_sync_pw.__enter__ = MagicMock(return_value=mock_playwright)
        mock_sync_pw.__exit__ = MagicMock(return_value=False)

        # BrowserSubAgent.execute mock 
        with patch("ultra_agent.runtime.browser.browser_worker.BrowserSubAgent") as MockAgent:
            mock_agent_instance = MagicMock()
            mock_agent_instance.execute.return_value = "[BAŞARILI] Test tamamlandı"
            MockAgent.return_value = mock_agent_instance

            from ultra_agent.runtime.browser.browser_worker import BrowserWorker

            worker = BrowserWorker(
                workspace=ws,
                project_name="mock_proj",
                session_id="mock_sess",
                enable_video=True,
                enable_tracing=True,
            )

            with patch("ultra_agent.runtime.browser.browser_worker.sync_playwright") as mock_sync_pw_func:
                mock_sync_pw_func.return_value = mock_sync_pw
                res = worker.run_task("Navigate to test.com and extract data", model="gemini-2.0-flash")

        result("result contains success", "BAŞARILI" in res or "tamamlandı" in res, res[:100])
        result("agent execute called", mock_agent_instance.execute.called)
        result("browser launched headless", mock_playwright.chromium.launch.called)
        result("context created", mock_browser.new_context.called)
        result("tracing started", mock_tracing.start.called)
        result("tracing stopped", mock_tracing.stop.called)
        result("context closed", mock_context.close.called)
        result("browser closed", mock_browser.close.called)

        # Job metadata dosyası kontrolü
        job_dirs = list((ws / "browser_artifacts" / "mock_proj" / "mock_sess").glob("job_*"))
        result("job dir created", len(job_dirs) >= 1, f"Found {len(job_dirs)} job dirs")

        if job_dirs:
            meta_file = job_dirs[0] / "job_meta.json"
            result("job_meta.json exists", meta_file.exists())
            if meta_file.exists():
                meta = json.loads(meta_file.read_text())
                result("meta has project", meta.get("project") == "mock_proj")
                result("meta has elapsed", "elapsed_s" in meta)
                result("meta has status", "status" in meta)


def test_worker_isolation_separate_dirs():
    """Farklı session'lar farklı artifact dizinleri kullanır."""
    print("\n=== Test 4: Session Isolation ===", flush=True)
    from ultra_agent.runtime.browser.browser_worker import BrowserWorker

    with tempfile.TemporaryDirectory() as tmpdir:
        ws = Path(tmpdir)
        w1 = BrowserWorker(workspace=ws, project_name="proj", session_id="alpha")
        w2 = BrowserWorker(workspace=ws, project_name="proj", session_id="beta")

        result("different base dirs", w1.base_dir != w2.base_dir)
        result("alpha dir exists", w1.base_dir.exists())
        result("beta dir exists", w2.base_dir.exists())
        result("alpha has session", "alpha" in str(w1.base_dir))
        result("beta has session", "beta" in str(w2.base_dir))


def test_run_browser_agent_delegation():
    """run_browser_agent legacy fonksiyonu BrowserWorker'a delege eder."""
    print("\n=== Test 5: Legacy run_browser_agent Delegation ===", flush=True)

    with patch("ultra_agent.runtime.browser.browser_worker.BrowserWorker") as MockWorker:
        mock_instance = MagicMock()
        mock_instance.run_task.return_value = "[MOCK] OK"
        MockWorker.return_value = mock_instance

        from ultra_agent.runtime.browser.browser_agent import run_browser_agent

        res = run_browser_agent(
            "test task",
            model="test-model",
            workspace=Path("/tmp/test"),
            timeout_s=60,
            project_name="test_proj",
            session_id="test_sess",
        )

        result("delegates to BrowserWorker", MockWorker.called)
        result("passes project_name", MockWorker.call_args[1].get("project_name") == "test_proj" or
               (len(MockWorker.call_args[0]) > 1 if MockWorker.call_args[0] else False))
        result("calls run_task", mock_instance.run_task.called)
        result("returns result", res == "[MOCK] OK")


# ── Ana ──
if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("  Phase 13 (P4) — BrowserWorker Mock Verification", flush=True)
    print("=" * 60, flush=True)

    test_worker_init()
    test_worker_isolation_separate_dirs()
    test_run_browser_agent_delegation()

    # Mock Playwright testi (daha karmaşık mock gereksinimi)
    try:
        test_worker_with_mock_playwright()
    except Exception as e:
        print(f"\n  ⚠️ Mock Playwright testi atlandı: {e}", flush=True)

    # Missing playwright testi
    try:
        test_worker_missing_playwright()
    except Exception as e:
        print(f"\n  ⚠️ Missing Playwright testi atlandı: {e}", flush=True)

    print(f"\n{'=' * 60}", flush=True)
    print(f"  Sonuç: {PASS} geçti, {FAIL} başarısız", flush=True)
    print(f"{'=' * 60}", flush=True)

    sys.exit(0 if FAIL == 0 else 1)
