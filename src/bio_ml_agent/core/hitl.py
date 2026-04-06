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

        # Üretim modunda burada Redis üzerinden API callback beklenecek
        import uuid
        import time
        import json

        approval_id = uuid.uuid4().hex
        log.warning(
            "⏸️ HITL: ONAY BEKLENİYOR | id=%s user=%s action=%s reason=%s",
            approval_id, user_id, action, reason or "N/A"
        )

        try:
            from bio_ml_agent.utils.config import get_config
            from redis import Redis
            cfg = get_config()
            redis_conn = Redis(
                host=cfg.redis.host,
                port=cfg.redis.port,
                db=cfg.redis.db,
                password=cfg.redis.password or None
            )

            req_key = f"hitl:request:{approval_id}"
            res_key = f"hitl:response:{approval_id}"

            # Kayıt atalım ki UI görebilsin
            req_data = {
                "id": approval_id,
                "user_id": user_id,
                "action": action,
                "details": details,
                "reason": reason,
                "timestamp": time.time()
            }
            redis_conn.setex(req_key, 3600, json.dumps(req_data)) # 1 saat geçerli

            log.info("Sistem %s Nolu Onay için API / WebSocket üzerinden bekliyor...", approval_id)

            # Wait for response (Polling)
            timeout = 300 # 5 dk bekleme süresi
            start = time.time()
            while time.time() - start < timeout:
                res = redis_conn.get(res_key)
                if res:
                    res_data = json.loads(res)
                    is_approved = res_data.get("approved", False)
                    log.info("✅ HITL Yanıtı Alındı | id=%s | onay=%s", approval_id, is_approved)

                    self._approval_log.append({
                        "id": approval_id,
                        "user_id": user_id,
                        "action": action,
                        "details": details,
                        "approved": is_approved,
                        "mode": "manual_redis",
                    })

                    redis_conn.delete(req_key)
                    redis_conn.delete(res_key)
                    return is_approved

                time.sleep(2)

            log.error("⏳ HITL Zaman Aşımı | id=%s", approval_id)
            redis_conn.delete(req_key)
            return False

        except Exception as e:
            log.error("HITL Redis bağlantı veya bekleme hatası: %s", e)
            return False

    def get_approval_log(self) -> list[Dict[str, Any]]:
        """Onay geçmişini döndür."""
        return list(self._approval_log)

