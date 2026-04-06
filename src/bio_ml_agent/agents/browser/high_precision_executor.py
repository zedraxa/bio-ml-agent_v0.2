import logging
import random
import asyncio
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

log = logging.getLogger("browser.precision_executor")

@dataclass
class ActionSnapshot:
    """Bir eylem öncesi sayfa durumunu saklar."""
    url: str
    timestamp: float
    screenshot_path: Optional[str] = None
    dom_summary: Optional[str] = None

class HighPrecisionExecutor:
    """
    HighPrecisionExecutor: Tarayıcı etkileşimlerini hatasız ve insansı (human-like) yapar.
    - SafeClick: Görünürlük, tıklanabilirlik ve çakışma (obscured) kontrolü.
    - HumanType: Karakterler arası rastgele gecikmeler.
    - RobustScroll: Lazy-loading ve sonsuz kaydırma desteği.
    - State Snapshots: Hata durumunda geri sarabilmek için durumu kaydeder.
    """

    def __init__(self, page: Any):
        self.page = page
        self.history: List[ActionSnapshot] = []

    async def take_snapshot(self, label: str) -> ActionSnapshot:
        """Geri dönüş (rollback) noktası oluşturur."""
        snapshot = ActionSnapshot(
            url=self.page.url,
            timestamp=time.time()
        )
        self.history.append(snapshot)
        log.debug(f"📸 Snapshot taken: {label}")
        return snapshot

    async def safe_click(self, selector: str, timeout: int = 5000) -> bool:
        """Elemanın üzerine odaklanır, görünürlüğünü doğrular ve tıklar."""
        try:
            # 1. Bekle ve görünürlüğü doğrula
            element = await self.page.wait_for_selector(selector, state="visible", timeout=timeout)
            if not element:
                log.error(f"❌ Element not found for click: {selector}")
                return False

            # 2. Scroll-to-view
            await element.scroll_into_view_if_needed()

            # 3. Micro-wait (insansı tepki)
            await asyncio.sleep(random.uniform(0.1, 0.4))

            # 4. Tıkla (ve intercept kontrolü)
            await element.click()
            log.info(f"🖱️ SafeClick Success: {selector}")
            return True
        except Exception as e:
            log.warning(f"⚠️ SafeClick Failed on {selector}: {e}")
            return False

    async def human_type(self, selector: str, text: str, delay_range: tuple = (50, 200)):
        """İnsansı yazma simülasyonu."""
        try:
            element = await self.page.wait_for_selector(selector, state="visible")
            if not element: return
            await element.click() # Odaklan

            for char in text:
                await self.page.keyboard.type(char)
                await asyncio.sleep(random.randint(*delay_range) / 1000.0)

            log.info(f"⌨️ HumanType Success on {selector}")
        except Exception as e:
            log.error(f"❌ HumanType Error: {e}")

    async def robust_scroll(self, direction: str = "down", amount: int = 500):
        """Kaydırma ve lazy-loading beklemesi."""
        try:
            if direction == "down":
                await self.page.mouse.wheel(0, amount)
            else:
                await self.page.mouse.wheel(0, -amount)

            # Lazy loading için bekle
            await asyncio.sleep(1.0)
            log.debug(f"📜 Scrolled {direction} by {amount}")
        except Exception as e:
            log.error(f"❌ Scroll Error: {e}")
