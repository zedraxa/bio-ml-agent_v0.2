"""
BrowserPolicy — Tarayıcı İzolasyon Politikası (P5)

Üç izolasyon modu tanımlar:
  ephemeral : Her job sıfır context (varsayılan, stateless)
  session   : Aynı proje/session içinde state korunur (cookies, localStorage)
  trusted   : Önceden kaydedilmiş login state yüklenir + domain whitelist

Politika dosyası: browser_artifacts/<proje>/<session>/policy.json
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

log = logging.getLogger("browser_policy")

# ─── Sabitler ───
VALID_MODES = ("ephemeral", "session", "trusted")
DEFAULT_MAX_TIMEOUT = 180
POLICY_FILENAME = "policy.json"
STATE_FILENAME = "storage_state.json"


@dataclass
class BrowserPolicy:
    """Tek bir proje/session çifti için tarayıcı izolasyon politikası."""

    mode: str = "ephemeral"
    allowed_domains: List[str] = field(default_factory=list)
    blocked_domains: List[str] = field(default_factory=list)
    max_timeout_s: int = DEFAULT_MAX_TIMEOUT
    enable_video: bool = True
    enable_tracing: bool = True
    storage_state_path: Optional[str] = None  # trusted mod: mutlak/göreli yol

    def __post_init__(self):
        if self.mode not in VALID_MODES:
            log.warning("⚠️ Geçersiz mod '%s', 'ephemeral' olarak düşürüldü", self.mode)
            self.mode = "ephemeral"

    # ── Kısayol sorguları ──

    @property
    def is_ephemeral(self) -> bool:
        return self.mode == "ephemeral"

    @property
    def is_session(self) -> bool:
        return self.mode == "session"

    @property
    def is_trusted(self) -> bool:
        return self.mode == "trusted"

    @property
    def should_persist_state(self) -> bool:
        """Context kapanırken state diske yazılmalı mı?"""
        return self.mode in ("session", "trusted")

    @property
    def should_load_state(self) -> bool:
        """Context açılırken mevcut state dosyası okunmalı mı?"""
        return self.mode in ("session", "trusted")

    # ── Domain kontrolü ──

    def is_domain_allowed(self, url: str) -> bool:
        """URL'nin politikaya uygun olup olmadığını kontrol eder."""
        from urllib.parse import urlparse
        try:
            domain = urlparse(url).netloc.lower()
        except Exception:
            return False

        # Bloklanmış domain kontrolü (öncelikli)
        for pat in self.blocked_domains:
            if pat in domain:
                return False

        # Whitelist varsa sadece izinliler geçer
        if self.allowed_domains:
            return any(pat in domain for pat in self.allowed_domains)

        return True  # Whitelist boşsa hepsi izinli

    # ── Seri/deseri ──

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "BrowserPolicy":
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    def save(self, directory: Path) -> Path:
        """Politikayı JSON dosyasına yazar."""
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / POLICY_FILENAME
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("📋 Politika kaydedildi: %s (mod=%s)", path, self.mode)
        return path

    @classmethod
    def load(cls, directory: Path) -> "BrowserPolicy":
        """Politika dosyasını okur; yoksa varsayılan döner."""
        path = directory / POLICY_FILENAME
        if not path.exists():
            log.info("📋 Politika dosyası bulunamadı, varsayılan (ephemeral) kullanılıyor: %s", path)
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            policy = cls.from_dict(data)
            log.info("📋 Politika yüklendi: %s (mod=%s)", path, policy.mode)
            return policy
        except Exception as e:
            log.warning("⚠️ Politika dosyası okunamadı (%s), varsayılan kullanılıyor: %s", path, e)
            return cls()


def resolve_state_path(policy: BrowserPolicy, base_dir: Path) -> Optional[Path]:
    """Politikaya göre storage_state dosyasının tam yolunu çözer."""
    if policy.is_ephemeral:
        return None

    if policy.storage_state_path:
        p = Path(policy.storage_state_path)
        if p.is_absolute():
            return p
        return base_dir / p

    # Varsayılan konum
    return base_dir / STATE_FILENAME
