# utils/config.py
# ═══════════════════════════════════════════════════════════
#  Bio-ML Agent — Yapılandırma Yönetim Modülü (Pydantic Destekli)
#
#  Öncelik sırası (yüksekten düşüğe):
#    1. Komut satırı argümanları (--model, --timeout vb.)
#    2. Ortam değişkenleri (OLLAMA_MODEL, AGENT_TIMEOUT vb.)
#    3. config.yaml dosyası
#    4. Varsayılan (default) değerler
# ═══════════════════════════════════════════════════════════

import os
import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ValidationError

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

log = logging.getLogger("bio_ml_agent")

# ─────────────────────────────────────────────
#  Pydantic Veri Modelleri
# ─────────────────────────────────────────────

class AgentSection(BaseModel):
    model: str = Field(default="qwen2.5:7b-instruct", description="Kullanılacak LLM modeli")
    max_steps: int = Field(default=50, ge=1, le=100)
    timeout: int = Field(default=180, ge=10, description="Araç (tool) çalışma zaman aşımı (saniye)")
    language: str = Field(default="tr")


class SecuritySection(BaseModel):
    allow_web_search: bool = Field(default=True)
    deny_patterns: List[str] = Field(default_factory=lambda: [
        r"\brm\b.*-rf\s+/",
        r":\(\)\s*{\s*:\s*\|\s*:\s*&\s*}\s*;\s*:",
        r"\bdd\b\s+if=/dev/zero\b",
        r"\bmkfs\.",
        r"\bshutdown\b",
        r"\breboot\b",
        r"\bkill\b\s+-9\s+1\b",
    ])


class WorkspaceConfig(BaseModel):
    default_project: str = Field(default="scratch_project")
    base_dir: str = Field(default="workspace")
    auto_save_web: bool = Field(default=True)


class HistoryConfig(BaseModel):
    directory: str = Field(default="conversation_history")
    auto_save_interval: int = Field(default=5, ge=1)
    max_summary_length: int = Field(default=100, ge=10)


class AgentConfig(BaseModel):
    model: str = Field(default="gpt-4o")
    max_steps: int = Field(default=30, ge=1, le=100)
    timeout: int = Field(default=300, ge=10, le=3600)
    language: str = Field(default="tr")

class SecurityConfig(BaseModel):
    api_key: str = Field(default="", description="API erişimi için gereken anahtar (boş ise güvenlik kapalıdır)")
    allow_web_search: bool = True
    deny_patterns: Optional[list[str]] = Field(
        default=None,
        description="Kara listedeki zararlı komut regex desenleri"
    )

class LoggingConfig(BaseModel):
    level: str = Field(default="INFO")
    directory: str = Field(default="logs")
    file_name: str = Field(default="agent.log")
    max_bytes: int = Field(default=10485760, ge=1048576)  # 10 MB
    backup_count: int = Field(default=5, ge=1)
    console_level: str = Field(default="WARNING")


class ComparisonConfig(BaseModel):
    enabled: bool = Field(default=True)
    generate_plots: bool = Field(default=True)
    plot_dpi: int = Field(default=150, ge=72, le=600)
    models: List[str] = Field(default_factory=lambda: ["Logistic Regression", "Random Forest", "SVM", "Gradient Boosting", "KNN"])


class MLConfig(BaseModel):
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    random_state: int = Field(default=42)
    cv_folds: int = Field(default=5, ge=2)
    default_task: str = Field(default="classification")
    comparison: ComparisonConfig = Field(default_factory=ComparisonConfig)

class RedisConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    password: str = ""
    db: int = 0

class AppConfig(BaseModel):
    """Ana yapılandırma sınıfı — tüm bölümleri 'nokta' notasyonuyla erişilebilir tutar."""
    agent: AgentConfig = Field(default_factory=AgentConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    history: HistoryConfig = Field(default_factory=HistoryConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    
    # Model objesinde _source alanına doğrudan izin verilmesi için model_config ekliyoruz
    # ya da objeye sonradan özellik olarak ekleriz.
    
    def to_dict(self) -> Dict[str, Any]:
        """Tüm yapılandırmayı dict olarak döndürür."""
        if hasattr(self, "model_dump"):
            return self.model_dump()
        return self.dict()

    def summary(self) -> str:
        """Yapılandırma özetini string olarak döndürür."""
        source_label = getattr(self, "_source", "defaults")
        lines = [
            f"📋 Yapılandırma (kaynak: {source_label})",
            f"   Model:        {self.agent.model}",
            f"   Maks Adım:    {self.agent.max_steps}",
            f"   Timeout:      {self.agent.timeout}s",
            f"   Dil:          {self.agent.language}",
            f"   Workspace:    {self.workspace.base_dir}",
            f"   Proje:        {self.workspace.default_project}",
            f"   Web Arama:    {'✅' if self.security.allow_web_search else '❌'}",
            f"   Log Seviyesi: {self.logging.level}",
            f"   ML CV Fold:   {self.ml.cv_folds}",
            f"   Karşılaştırma:{' ✅' if self.ml.comparison.enabled else '❌'}",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────
#  YAML Yükleme ve Merge Fonksiyonları
# ─────────────────────────────────────────────

def _deep_merge(base: Dict, override: Dict) -> Dict:
    """İki dict'i derin birleştirir. Override'daki değerler base'in üzerine yazılır."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_yaml_file(path: Path) -> Dict[str, Any]:
    """YAML dosyasını yükler."""
    if not path.exists():
        log.debug("Config dosyası bulunamadı: %s", path)
        return {}
    if yaml is None:
        log.warning("PyYAML yüklü değil. config.yaml okunamıyor.")
        return {}
    try:
        content = path.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        if not isinstance(data, dict):
            log.warning("config.yaml geçersiz format — dict bekleniyor.")
            return {}
        return data
    except Exception as e:
        log.error("config.yaml okuma hatası: %s", e)
        return {}


def _apply_env_overrides(data: Dict[str, Any]) -> Dict[str, Any]:
    """Ortam değişkenlerinden yapılandırma değerlerini override eder."""
    env_map = [
        ("OLLAMA_MODEL",       "agent",     "model",            str),
        ("AGENT_TIMEOUT",      "agent",     "timeout",          int),
        ("AGENT_MAX_STEPS",    "agent",     "max_steps",        int),
        ("AGENT_LANGUAGE",     "agent",     "language",          str),
        ("AGENT_WORKSPACE",    "workspace", "base_dir",         str),
        ("AGENT_HISTORY_DIR",  "history",   "directory",         str),
        ("AGENT_LOG_LEVEL",    "logging",   "level",            str),
        ("AGENT_LOG_DIR",      "logging",   "directory",         str),
        ("AGENT_WEB_SEARCH",   "security",  "allow_web_search", lambda x: x.lower() in ("true", "1", "yes")),
        ("AGENT_API_KEY",      "security",  "api_key",          str),
        ("REDIS_HOST",         "redis",     "host",             str),
        ("REDIS_PORT",         "redis",     "port",             int),
    ]

    for env_var, section, key, conv in env_map:
        val = os.environ.get(env_var)
        if val is not None:
            if section not in data:
                data[section] = {}
            try:
                data[section][key] = conv(val)
                log.debug("ENV override: %s → %s.%s = %s", env_var, section, key, val)
            except (ValueError, TypeError) as e:
                log.warning("ENV %s geçersiz değer '%s': %s", env_var, val, e)

    return data


# ─────────────────────────────────────────────
#  Ana API
# ─────────────────────────────────────────────

_config: Optional[AppConfig] = None


def load_config(
    config_path: Optional[str] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> AppConfig:
    """Yapılandırmayı Pydantic model kurallarına uygun olarak yükler ve doğrular."""
    global _config

    # Varsayılan konfig objesini dummy dict'le başlatıp dump ederek tree'yi alalım
    temp_config = AppConfig()
    if hasattr(temp_config, "model_dump"):
        base_dict = temp_config.model_dump()
    else:
        base_dict = temp_config.dict()
    
    if config_path is None:
        candidates = [
            Path("config.yaml"),
            Path("config.yml"),
            Path(__file__).parent.parent / "config.yaml",
            Path(__file__).parent.parent / "config.yml",
        ]
        for candidate in candidates:
            if candidate.exists():
                config_path = str(candidate)
                break

    source = "defaults"
    if config_path:
        yaml_data = _load_yaml_file(Path(config_path))
        if yaml_data:
            base_dict = _deep_merge(base_dict, yaml_data)
            source = config_path

    base_dict = _apply_env_overrides(base_dict)
    if any(os.environ.get(k) for k in [
        "OLLAMA_MODEL", "AGENT_TIMEOUT", "AGENT_MAX_STEPS",
        "AGENT_WORKSPACE", "AGENT_LOG_LEVEL"
    ]):
        source += " + env"

    if cli_overrides:
        base_dict = _deep_merge(base_dict, cli_overrides)
        source += " + cli"

    # Pydantic validasyonu! Bu aşamada hatalı (Örn step=-1) veri girildiyse crash verir.
    try:
        _config = AppConfig(**base_dict)
    except ValidationError as e:
        log.error("Konfigürasyon doğrulama hatası! Lütfen config.yaml ve ENV değişkenlerinizi kontrol edin.")
        log.error(e)
        raise e

    try:
        object.__setattr__(_config, '_source', source)
    except Exception:
        pass
    return _config


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config() -> None:
    global _config
    _config = None
