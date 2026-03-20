"""
core.hitl — Human-in-the-Loop (HITL) Onay Mekanizması

Otomatik agent aksiyonlarının insan onayı gerektiren kritik durumlarda
engellenmesini ve izlenmesini sağlar.

Kullanım:
    from bio_ml_agent.core.hitl import HITLManager
    hitl = HITLManager(workspace)
    is_ok = hitl.require_approval(user_id, action, details)
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger("bio_ml_agent")


class HITLPolicy:
    """Hangi aksiyonların onay gerektirdiğini tanımlayan politika."""

    # Varsayılan olarak onay gerektiren aksiyonlar
    REQUIRE_APPROVAL = {
        "BASH_EXEC",
        "DELETE_FILE",
        "DEPLOY",
        "INSTALL_PACKAGE",
        "EXTERNAL_API_CALL",
    }

    # Auto-approve edilen aksiyonlar (güvenli)
    AUTO_APPROVE = {
        "READ_FILE",
        "LIST_DIR",
        "PYTHON_EXEC_SAFE",
        "WEB_SEARCH",
    }

    @classmethod
    def needs_approval(cls, action: str) -> bool:
        """Bir aksiyonun onay gerektirip gerektirmediğini döndür."""
        if action in cls.AUTO_APPROVE:
            return False
        if action in cls.REQUIRE_APPROVAL:
            return True
        # Bilinmeyen aksiyonlar için güvenli tarafta kal
        return True


class HITLManager:
    """
    Human-in-the-Loop yöneticisi.

    Üretim ortamında bu sınıf UI üzerinden onay isteğini gösterir.
    Geliştirme ortamında varsayılan olarak otomatik onay verir.
    """

    def __init__(self, workspace: Path, auto_approve: bool = True):
        """
        Args:
            workspace: Çalışma dizini
            auto_approve: True ise geliştirme modunda otomatik onay ver.
                          Üretim ortamında False olmalıdır.
        """
        self.workspace = workspace
        self.auto_approve = auto_approve
        self._approval_log: list[Dict[str, Any]] = []

    def require_approval(
        self,
        user_id: str,
        action: str,
        details: Dict[str, Any],
        reason: Optional[str] = None,
    ) -> bool:
        """Bir aksiyon için onay iste.

        Args:
            user_id: İsteği yapan kullanıcı/agent kimliği
            action: Aksiyonun türü (BASH_EXEC, DELETE_FILE, vb.)
            details: Aksiyonun detayları
            reason: Opsiyonel neden açıklaması

        Returns:
            True: Onaylandı, False: Reddedildi
        """
        needs = HITLPolicy.needs_approval(action)

        if not needs:
            log.debug("✅ HITL: Otomatik onay (güvenli aksiyon) | action=%s", action)
            return True

        if self.auto_approve:
            log.info(
                "✅ HITL: Dev-mode otomatik onay | user=%s action=%s details=%s",
                user_id, action, str(details)[:200]
            )
            self._approval_log.append({
                "user_id": user_id,
                "action": action,
                "details": details,
                "approved": True,
                "mode": "auto_approve",
            })
            return True

        # Üretim modunda burada UI/CLI üzerinden onay istenecek
        log.warning(
            "⏸️ HITL: ONAY BEKLENİYOR | user=%s action=%s reason=%s",
            user_id, action, reason or "N/A"
        )
        # TODO: Üretim ortamında WebSocket veya API callback ile bekleme
        return False

    def get_approval_log(self) -> list[Dict[str, Any]]:
        """Onay geçmişini döndür."""
        return list(self._approval_log)
