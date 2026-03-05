# core/config.py
# ═══════════════════════════════════════════════════════════
#  Merkezi yapılandırma: AgentConfig + SYSTEM_PROMPT
#  Faz 2 — agent.py'den taşındı
# ═══════════════════════════════════════════════════════════

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Agent çalışma yapılandırması."""
    model: str = Field(default="qwen2.5:7b-instruct")
    workspace: Path = Field(default=Path("workspace"))
    timeout: int = Field(default=180)
    max_steps: int = Field(default=9999)
    history_dir: Path = Field(default=Path("conversation_history"))
    load_session: Optional[str] = None
    log_level: str = "INFO"
    log_dir: Path = Field(default=Path("logs"))
    config_file: str = "config.yaml"
    backend_mode: str = "auto"
    swarm: bool = Field(default=False)


# ─────────────────────────────────────────────
#  SYSTEM_PROMPT — dosyadan yüklenir
# ─────────────────────────────────────────────

_PROMPT_FILE = Path(__file__).resolve().parent / "system_prompt.txt"
SYSTEM_PROMPT = _PROMPT_FILE.read_text(encoding="utf-8") if _PROMPT_FILE.exists() else "You are a Bioengineering ML Agent."
