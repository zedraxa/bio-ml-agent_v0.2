import logging
import time
from typing import List, Dict, Any, Optional

log = logging.getLogger("browser.autopilot")

class BrowserAutopilot:
    """
    BrowserAutopilot: Tarayıcıdaki "gürültüyü" temizler.
    - Cookie banner'larını tespit eder ve kapatır.
    - Pop-up ve overlay'leri (interstitial) yakalar.
    - Sayfanın etkileşime hazır olmasını sağlar.
    """

    def __init__(self, page: Any):
        self.page = page

    async def cleanup(self):
        """Sayfayı temizlemek için bir dizi otonom adım atar."""
        log.info("🧹 Starting page cleanup...")

        # 1. Yaygın cookie banner butonlarını ara
        selectors = [
            "button:has-text('Accept')", "button:has-text('Allow')",
            "button:has-text('Agree')", "button:has-text('Çerezleri Kabul Et')",
            "#onetrust-accept-btn-handler", ".cookie-accept-button"
        ]

        for selector in selectors:
            try:
                # 2 saniye bekle ve varsa tıkla
                element = await self.page.wait_for_selector(selector, timeout=2000)
                if element:
                    await element.click()
                    log.info(f"✅ Clicked cookie banner: {selector}")
                    await self.page.wait_for_timeout(500)
            except Exception:
                continue

        # 2. Overlay'leri gizle (Z-index kontrolü veya ESC tuşu)
        try:
            await self.page.keyboard.press("Escape")
        except Exception:
            pass

        log.info("✨ Cleanup finished.")
