import logging
import time
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
        
        try:
            from playwright.sync_api import sync_playwright
            
            with sync_playwright() as p:
                browser = p.chromium.launch_persistent_context(
                    user_data_dir=str(self.profile_dir),
                    headless=True,
                    record_video_dir=str(self.recording_dir), # S7-4: Browser Recording
                    viewport={"width": 1280, "height": 720}
                )
                page = browser.new_page()
                page.goto(url, wait_until="networkidle", timeout=30000)
                
                # S7-2: Vision Fallback Screenshot (Screenshot doğrulaması)
                screenshot_path = self.recording_dir / f"vision_fb_{int(time.time())}.png"
                page.screenshot(path=str(screenshot_path))
                
                # S7-1: Temel DOM-first çıkarma
                text_content = page.evaluate("() => document.body.innerText")
                browser.close()
                log.info(f"Screenshot taken: {screenshot_path}")
                return text_content
                
        except ImportError:
            log.error("Playwright modülü mevcut değil. Lütfen pip install playwright kullanın.")
            return "ERROR: Web driver import failed."
        except Exception as e:
            log.error(f"Browser otomasyon hatası: {e}")
            return f"ERROR: {str(e)}"
