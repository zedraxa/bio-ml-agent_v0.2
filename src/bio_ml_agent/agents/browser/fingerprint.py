import logging
from typing import List, Dict, Any, Optional

log = logging.getLogger("browser.fingerprint")

class FingerprintDetector:
    """
    FingerprintDetector: Sayfadaki anti-bot mekanizmalarını tespit eder.
    - Cloudflare (Turnstile, Challenge).
    - Datadome, Akamai, PerimeterX belirteçleri.
    - "Bot detected", "Access Denied", "Verify you are human" metinleri.
    """

    @staticmethod
    def detect_risks(page_content: str, title: str) -> Dict[str, Any]:
        risks = {
            "cloudflare": False,
            "datadome": False,
            "bot_challenge": False,
            "access_denied": False
        }

        content_lower = page_content.lower()
        title_lower = title.lower()

        # Cloudflare Detection
        if "cloudflare" in content_lower or "turnstile" in content_lower:
            risks["cloudflare"] = True

        # Challenge / Human Verification
        if "verify you are human" in content_lower or "hcaptcha" in content_lower:
            risks["bot_challenge"] = True

        # Access Denied
        if "access denied" in title_lower or "403 forbidden" in content_lower:
            risks["access_denied"] = True

        return risks

    @staticmethod
    def get_risk_score(risks: Dict[str, Any]) -> float:
        """0.0 (Güvenli) ile 1.0 (Bloklanmış) arası bir skor döner."""
        score = 0.0
        if risks["access_denied"]: score += 0.9
        if risks["bot_challenge"]: score += 0.7
        if risks["cloudflare"]: score += 0.3
        return min(score, 1.0)
