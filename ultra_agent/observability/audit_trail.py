import logging
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

log = logging.getLogger("bio_ml_agent")

class AuditTrailLogger:
    """
    S8-3 & S8-4: Governance ve Audit Trails (Denetim İzi).
    Kritik eylemleri (Örn. dosya silme, container başlatma, yetki isteme)
    değiştirilemez/izole bir denetim günlüğüne yazar.
    Manuel Approval Gates gerektiren araçlar burada kayıt altına alınır.
    """
    def __init__(self, workspace: Path):
        self.audit_dir = workspace / "audit_logs"
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.audit_file = self.audit_dir / f"audit_{datetime.now().strftime('%Y%m')}.jsonl"
        
    def log_critical_action(self, agent_id: str, action: str, details: Dict[str, Any], approval_status: str):
        """
        S8-4: 5 kritik tool operasyonunda (Approval Gates) çalışması hedeflenir.
        Örn: [BASH_ROOT, DELETE_FILE, REGISTER_API, SEND_EMAIL, EXEC_PLUGIN]
        """
        entry = {
            "timestamp": datetime.now().isoformat() + "Z",
            "agent_id": agent_id,
            "action": action,
            "approval_status": approval_status,
            "details": details,
            "user": os.getenv("USER", "unknown_user")
        }
        
        with open(self.audit_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
            
        log.info(f"🔒 AUDIT TRAIL: {action} logged. Status: {approval_status}")
