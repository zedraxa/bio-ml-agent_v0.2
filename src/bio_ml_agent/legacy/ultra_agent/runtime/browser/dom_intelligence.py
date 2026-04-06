"""
DOM Intelligence Katmanı (P6)

Tarayıcı agent'ının (BrowserSubAgent) karar mekanizmasını güçlendirir.
1. LocatorResolver: `bio-id` kimliklerini Playwright'ın en stabil semantik (text/role) lokasyonlarına çevirir.
2. ActionValidator: Aksiyon öncesinde DOM üzerinde güvenlik ve görünürlük kontrolleri yapar.
"""

import logging
from typing import Optional, Dict, Any, Tuple

# Playwright Locator type hint if available
try:
    from playwright.sync_api import Locator
except ImportError:
    Locator = Any

log = logging.getLogger("dom_intelligence")


class LocatorResolver:
    """bio_id kimliklerini analiz edip en doğru Playwright Locator'ı döndürür."""

    def __init__(self, page: Any, perception_data: Dict[str, Any]):
        self.page = page
        self.perception = perception_data

    def _find_element_data(self, bio_id: str) -> Optional[Dict]:
        """Perception nesnesinden bio_id verisini bulur."""
        interactive = self.perception.get("interactive", [])
        for el in interactive:
            if str(el.get("bio_id")) == str(bio_id):
                return el
        return None

    def resolve(self, bio_id: str) -> Locator:
        """
        DOM verisine bakarak locator üretir. 
        Sıralama: 1. data-bio-id (Tüm frame'lerde arar) -> 2. Role+Name -> 3. Text
        """
        el_data = self._find_element_data(bio_id)

        # Strateji 0: data-bio-id ile tüm frame'lerde ara (En sağlamı)
        # Çünkü distiller.js bu özelliği tüm frame'lerdeki elemanlara bastı.
        for frame in self.page.frames:
            try:
                loc = frame.locator(f"[data-bio-id='{bio_id}']")
                if loc.count() > 0:
                    log.debug(f"🔍 Resolved {bio_id} using data-bio-id in a frame.")
                    return loc.first
            except Exception:
                continue

        if not el_data:
            log.warning(f"⚠️ bio_id '{bio_id}' DOM verisinde bulunamadı ve hiçbir frame'de eşleşmedi.")
            return self.page.locator(f"[data-bio-id='{bio_id}']").first

        pw_role = el_data.get("pw_role")
        pw_name = el_data.get("pw_name")
        text_content = el_data.get("text", "").strip()

        # Semantik stratejiler (Sadece ana frame'de kalabilir veya frames içinde denenebilir)
        # Şimdilik ana frame fallback olarak kalsın.
        try:
            if pw_role and pw_name and len(pw_name) > 1:
                loc = self.page.get_by_role(pw_role, name=pw_name)
                if loc.count() > 0: return loc.first

            if text_content and len(text_content) > 3:
                loc = self.page.get_by_text(text_content)
                if loc.count() > 0: return loc.first
        except Exception:
            pass

        return self.page.locator(f"[data-bio-id='{bio_id}']").first


class ActionValidator:
    """Aksiyonun geçerliliğini (görünürlük, tıklanabilirlik) kontrol eder."""

    def __init__(self, locator: Locator, action_type: str):
        self.locator = locator
        self.action_type = action_type

    def validate(self) -> Tuple[bool, str]:
        """
        Kontroller:
        1. Eşleşme var mı?
        2. Görünür mü?
        3. Disabled mı?
        Returns: (Is_Valid, Error_Reason)
        """
        try:
            count = self.locator.count()
            if count == 0:
                return False, "Locator DOM'da eşleşmedi (element kaybolmuş olabilir)."

            # Click, Fill vb. etkileşimli olaylar için görünürlük ve disabled kontrolü
            interactive_actions = ["click", "fill", "select", "press"]
            if self.action_type in interactive_actions:
                # 1. Görünürlük
                if not self.locator.first.is_visible(timeout=1000):
                    return False, "Element görünür değil (is_visible=False)."

                # 2. Disabled
                if self.locator.first.is_disabled(timeout=1000):
                    return False, "Element kullanılamaz durumda (disabled)."

                # 3. Editable (Fill için)
                if self.action_type == "fill":
                    if not self.locator.first.is_editable(timeout=1000):
                        return False, "Element yazılabilir/düzenlenebilir değil (editable=False)."

            return True, "Valid"

        except Exception as e:
            return False, f"Validasyon sırasında yakalanamayan hata: {str(e)}"
