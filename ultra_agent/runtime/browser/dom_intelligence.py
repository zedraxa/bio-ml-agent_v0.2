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
        Sıralama: 1. Role+Name -> 2. Text -> 3. Fallback CSS (data-bio-id)
        """
        el_data = self._find_element_data(bio_id)
        fallback_css = f"[data-bio-id='{bio_id}']"

        if not el_data:
            log.warning(f"⚠️ bio_id '{bio_id}' DOM verisinde bulunamadı. Fallback CSS kullanılıyor.")
            return self.page.locator(fallback_css).first

        pw_role = el_data.get("pw_role")
        pw_name = el_data.get("pw_name")
        text_content = el_data.get("text", "").strip()

        try:
            # Strateji 1: get_by_role (Semantic)
            if pw_role and pw_name and len(pw_name) > 1:
                loc = self.page.get_by_role(pw_role, name=pw_name)
                # Disambiguation: eğer birden çok eşleşme olursa tam eşleşme (exact) dene
                if loc.count() > 1:
                    loc_exact = self.page.get_by_role(pw_role, name=pw_name, exact=True)
                    if loc_exact.count() > 0:
                        loc = loc_exact
                
                # Hala birden çoksa veya çalıştıysa
                if loc.count() > 0:
                    log.debug(f"🔍 Resolved {bio_id} using Role: {pw_role}, Name: {pw_name}")
                    return loc.first

            # Strateji 2: get_by_text
            if text_content and len(text_content) > 3:
                loc = self.page.get_by_text(text_content)
                if loc.count() > 1:
                    loc_exact = self.page.get_by_text(text_content, exact=True)
                    if loc_exact.count() > 0:
                        loc = loc_exact
                
                if loc.count() > 0:
                    log.debug(f"🔍 Resolved {bio_id} using Text: {text_content[:20]}")
                    return loc.first

        except Exception as e:
            log.debug(f"⚠️ Resolution fallible for {bio_id}, err: {e}")

        # Strateji 3: Fallback (Yine de en garantili yol data-bio-id)
        log.debug(f"🔍 Resolved {bio_id} using Fallback CSS: {fallback_css}")
        return self.page.locator(fallback_css).first


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
