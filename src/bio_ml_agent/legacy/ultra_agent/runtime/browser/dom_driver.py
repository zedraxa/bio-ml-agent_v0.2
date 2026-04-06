import logging
import time
import json
from pathlib import Path
from typing import Optional

log = logging.getLogger("bio_ml_agent")

class DOMDriver:
    """
    S7-1: Playwright/Browser-Use tabanlı, DOM-first Browser Driver.
    S7-2: Vision Fallback ve Screenshot alma 
    S7-3: Tenant Session (izolasyon)
    S7-4: Recording Artifacts
    """
    def __init__(self, tenant_id: str, workspace_dir: Path):
        self.tenant_id = tenant_id
        # S7-3 İzole profile directory
        self.profile_dir = workspace_dir / "browser_profiles" / tenant_id
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        # S7-4 Artifact Directory
        self.recording_dir = workspace_dir / "artifacts" / "recordings"
        self.recording_dir.mkdir(parents=True, exist_ok=True)

    def navigate_and_extract(self, url: str) -> str:
        log.info(f"Navigate to {url} (Tenant: {self.tenant_id})")

        # Distiller scriptini yükle
        distiller_path = Path(__file__).parent / "distiller.js"
        distiller_js = distiller_path.read_text(encoding="utf-8") if distiller_path.exists() else "return {page: {}, interactive: []}"

        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True,
                    executable_path=None # Playwright handles this
                )
                context = browser.new_context(
                    user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
                    record_video_dir=str(self.recording_dir),
                    viewport={"width": 1280, "height": 720}
                )
                page = context.new_page()
                page.goto(url, wait_until="networkidle", timeout=30000)

                # S7-2: Vision Fallback Screenshot
                screenshot_path = self.recording_dir / f"vision_fb_{int(time.time())}.png"
                page.screenshot(path=str(screenshot_path))

                # P1: DOM Perception
                perception = page.evaluate(distiller_js)
                context.close()
                browser.close()
                log.info(f"Screenshot taken: {screenshot_path}")

                return json.dumps(perception, indent=2, ensure_ascii=False)

        except ImportError:
            log.error("Playwright modülü mevcut değil. Lütfen pip install playwright kullanın.")
            return "ERROR: Web driver import failed."
        except Exception as e:
            log.error(f"Browser otomasyon hatası: {e}")
            return f"ERROR: {str(e)}"
