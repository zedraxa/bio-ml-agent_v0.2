# utils/config.py
# ═══════════════════════════════════════════════════════════
#  Bio-ML Agent — Yapılandırma Yönetim Modülü
#
#  Öncelik sırası (yüksekten düşüğe):
#    1. Komut satırı argümanları (--model, --timeout vb.)
#    2. Ortam değişkenleri (OLLAMA_MODEL, AGENT_TIMEOUT vb.)
#    3. config.yaml dosyası
#    4. Varsayılan (default) değerler
#
#  Kullanım:
#    from utils.config import load_config, get_config
#    cfg = load_config("config.yaml")
#    print(cfg.agent.model)
# ═══════════════════════════════════════════════════════════

import os
import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

log = logging.getLogger("bio_ml_agent")


# ─────────────────────────────────────────────
#  Varsayılan Yapılandırma
# ─────────────────────────────────────────────

DEFAULTS: Dict[str, Any] = {
    "agent": {
        "model": "qwen2.5:7b-instruct",
        "max_steps": 50,
        "timeout": 180,
        "language": "tr",
    },
    "security": {
        "allow_web_search": False,
        "deny_patterns": [
            r"\brm\b.*-rf\s+/",
            r":\(\)\s*{\s*:\s*\|\s*:\s*&\s*}\s*;\s*:",
            r"\bdd\b\s+if=/dev/zero\b",
            r"\bmkfs\.",
            r"\bshutdown\b",
            r"\breboot\b",
            r"\bkill\b\s+-9\s+1\b",
        ],
    },
    "workspace": {
        "default_project": "scratch_project",
        "base_dir": "workspace",
        "auto_save_web": True,
    },
    "history": {
        "directory": "conversation_history",
        "auto_save_interval": 5,
        "max_summary_length": 100,
    },
    "logging": {
        "level": "INFO",
        "directory": "logs",
        "file_name": "agent.log",
        "max_bytes": 5 * 1024 * 1024,
        "backup_count": 3,
        "console_level": "WARNING",
    },
    "ml": {
        "test_size": 0.2,
        "random_state": 42,
        "cv_folds": 5,
        "default_task": "classification",
        "comparison": {
            "enabled": True,
            "generate_plots": True,
            "plot_dpi": 150,
            "output_formats": ["json", "csv", "markdown"],
        },
    },
}


# ─────────────────────────────────────────────
#  Yapılandırma Veri Sınıfları
# ─────────────────────────────────────────────

@dataclass
class AgentSection:
    """Agent ana ayarları."""
    model: str = "qwen2.5:7b-instruct"
    max_steps: int = 50
    timeout: int = 180
    language: str = "tr"


@dataclass
class SecuritySection:
    """Güvenlik ayarları."""
    allow_web_search: bool = False
    deny_patterns: List[str] = field(default_factory=lambda: list(DEFAULTS["security"]["deny_patterns"]))


@dataclass
class WorkspaceSection:
    """Çalışma alanı ayarları."""
    default_project: str = "scratch_project"
    base_dir: str = "workspace"
    auto_save_web: bool = True


@dataclass
class HistorySection:
    """Konuşma geçmişi ayarları."""
    directory: str = "conversation_history"
    auto_save_interval: int = 5
    max_summary_length: int = 100


@dataclass
class LoggingSection:
    """Loglama ayarları."""
    level: str = "INFO"
    directory: str = "logs"
    file_name: str = "agent.log"
    max_bytes: int = 5 * 1024 * 1024
    backup_count: int = 3
    console_level: str = "WARNING"


@dataclass
class ComparisonSection:
    """ML model karşılaştırma ayarları."""
    enabled: bool = True
    generate_plots: bool = True
    plot_dpi: int = 150
    output_formats: List[str] = field(default_factory=lambda: ["json", "csv", "markdown"])


@dataclass
class MLSection:
    """ML ayarları."""
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    default_task: str = "classification"
    comparison: ComparisonSection = field(default_factory=ComparisonSection)


@dataclass
class Config:
    """Ana yapılandırma sınıfı — tüm bölümleri 'nokta' notasyonuyla erişilebilir tutar."""
    agent: AgentSection = field(default_factory=AgentSection)
    security: SecuritySection = field(default_factory=SecuritySection)
    workspace: WorkspaceSection = field(default_factory=WorkspaceSection)
    history: HistorySection = field(default_factory=HistorySection)
    logging: LoggingSection = field(default_factory=LoggingSection)
    ml: MLSection = field(default_factory=MLSection)

    # Kaynak dosya (debug amaçlı)
    _source: str = "defaults"

    def to_dict(self) -> Dict[str, Any]:
        """Tüm yapılandırmayı dict olarak döndürür."""
        from dataclasses import asdict
        d = asdict(self)
        d.pop("_source", None)
        return d

    def summary(self) -> str:
        """Yapılandırma özetini string olarak döndürür."""
        lines = [
            f"📋 Yapılandırma (kaynak: {self._source})",
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
#  YAML Yükleme Fonksiyonları
# ─────────────────────────────────────────────

def _deep_merge(base: Dict, override: Dict) -> Dict:
    """
    İki dict'i derin birleştirir. Override'daki değerler base'in üzerine yazılır.
    Alt dict'ler recursif olarak birleştirilir.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_yaml_file(path: Path) -> Dict[str, Any]:
    """YAML dosyasını yükler. yaml modülü yoksa boş dict döner."""
    if not path.exists():
        log.debug("Config dosyası bulunamadı: %s", path)
        return {}

    if yaml is None:
        log.warning("PyYAML yüklü değil. config.yaml okunamıyor. "
                     "Yüklemek için: pip install pyyaml")
        return {}

    try:
        content = path.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        if not isinstance(data, dict):
            log.warning("config.yaml geçersiz format — dict bekleniyor, %s alındı", type(data).__name__)
            return {}
        log.info("📋 Config dosyası yüklendi: %s", path)
        return data
    except yaml.YAMLError as e:
        log.error("config.yaml parse hatası: %s", e)
        return {}
    except Exception as e:
        log.error("config.yaml okuma hatası: %s", e)
        return {}


def _apply_env_overrides(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ortam değişkenlerinden yapılandırma değerlerini override eder.

    Desteklenen ortam değişkenleri:
      OLLAMA_MODEL       → agent.model
      AGENT_TIMEOUT      → agent.timeout
      AGENT_MAX_STEPS    → agent.max_steps
      AGENT_LANGUAGE     → agent.language
      AGENT_WORKSPACE    → workspace.base_dir
      AGENT_HISTORY_DIR  → history.directory
      AGENT_LOG_LEVEL    → logging.level
      AGENT_LOG_DIR      → logging.directory
      AGENT_WEB_SEARCH   → security.allow_web_search
    """
    env_map = [
        # (env_var, section, key, type_converter)
        ("OLLAMA_MODEL",       "agent",     "model",            str),
        ("AGENT_TIMEOUT",      "agent",     "timeout",          int),
        ("AGENT_MAX_STEPS",    "agent",     "max_steps",        int),
        ("AGENT_LANGUAGE",     "agent",     "language",          str),
        ("AGENT_WORKSPACE",    "workspace", "base_dir",         str),
        ("AGENT_HISTORY_DIR",  "history",   "directory",         str),
        ("AGENT_LOG_LEVEL",    "logging",   "level",            str),
        ("AGENT_LOG_DIR",      "logging",   "directory",         str),
        ("AGENT_WEB_SEARCH",   "security",  "allow_web_search", lambda x: x.lower() in ("true", "1", "yes")),
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


def _dict_to_config(data: Dict[str, Any]) -> Config:
    """Dict'i Config dataclass'a dönüştürür."""
    cfg = Config()

    # Agent
    agent_data = data.get("agent", {})
    cfg.agent = AgentSection(
        model=agent_data.get("model", cfg.agent.model),
        max_steps=int(agent_data.get("max_steps", cfg.agent.max_steps)),
        timeout=int(agent_data.get("timeout", cfg.agent.timeout)),
        language=str(agent_data.get("language", cfg.agent.language)),
    )

    # Security
    sec_data = data.get("security", {})
    cfg.security = SecuritySection(
        allow_web_search=bool(sec_data.get("allow_web_search", cfg.security.allow_web_search)),
        deny_patterns=sec_data.get("deny_patterns", cfg.security.deny_patterns),
    )

    # Workspace
    ws_data = data.get("workspace", {})
    cfg.workspace = WorkspaceSection(
        default_project=str(ws_data.get("default_project", cfg.workspace.default_project)),
        base_dir=str(ws_data.get("base_dir", cfg.workspace.base_dir)),
        auto_save_web=bool(ws_data.get("auto_save_web", cfg.workspace.auto_save_web)),
    )

    # History
    hist_data = data.get("history", {})
    cfg.history = HistorySection(
        directory=str(hist_data.get("directory", cfg.history.directory)),
        auto_save_interval=int(hist_data.get("auto_save_interval", cfg.history.auto_save_interval)),
        max_summary_length=int(hist_data.get("max_summary_length", cfg.history.max_summary_length)),
    )

    # Logging
    log_data = data.get("logging", {})
    cfg.logging = LoggingSection(
        level=str(log_data.get("level", cfg.logging.level)),
        directory=str(log_data.get("directory", cfg.logging.directory)),
        file_name=str(log_data.get("file_name", cfg.logging.file_name)),
        max_bytes=int(log_data.get("max_bytes", cfg.logging.max_bytes)),
        backup_count=int(log_data.get("backup_count", cfg.logging.backup_count)),
        console_level=str(log_data.get("console_level", cfg.logging.console_level)),
    )

    # ML
    ml_data = data.get("ml", {})
    comp_data = ml_data.get("comparison", {})
    cfg.ml = MLSection(
        test_size=float(ml_data.get("test_size", cfg.ml.test_size)),
        random_state=int(ml_data.get("random_state", cfg.ml.random_state)),
        cv_folds=int(ml_data.get("cv_folds", cfg.ml.cv_folds)),
        default_task=str(ml_data.get("default_task", cfg.ml.default_task)),
        comparison=ComparisonSection(
            enabled=bool(comp_data.get("enabled", cfg.ml.comparison.enabled)),
            generate_plots=bool(comp_data.get("generate_plots", cfg.ml.comparison.generate_plots)),
            plot_dpi=int(comp_data.get("plot_dpi", cfg.ml.comparison.plot_dpi)),
            output_formats=comp_data.get("output_formats", cfg.ml.comparison.output_formats),
        ),
    )

    return cfg


# ─────────────────────────────────────────────
#  Ana API
# ─────────────────────────────────────────────

# Global config instance
_config: Optional[Config] = None


def load_config(
    config_path: Optional[str] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    """
    Yapılandırmayı yükler. Öncelik sırası:
      1. cli_overrides (komut satırı argümanları)
      2. Ortam değişkenleri
      3. config.yaml dosyası
      4. Varsayılan (DEFAULTS) değerler

    Args:
        config_path: YAML dosya yolu (None ise config.yaml aranır)
        cli_overrides: Komut satırı argümanlarından gelen override'lar

    Returns:
        Config nesnesi
    """
    global _config

    # 1. Varsayılanlarla başla
    merged = copy.deepcopy(DEFAULTS)

    # 2. YAML dosyasını yükle ve birleştir
    if config_path is None:
        # Proje kök dizininde config.yaml ara
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
            merged = _deep_merge(merged, yaml_data)
            source = config_path

    # 3. Ortam değişkenlerini uygula
    merged = _apply_env_overrides(merged)
    if any(os.environ.get(k) for k in [
        "OLLAMA_MODEL", "AGENT_TIMEOUT", "AGENT_MAX_STEPS",
        "AGENT_WORKSPACE", "AGENT_LOG_LEVEL"
    ]):
        source += " + env"

    # 4. CLI override'ları uygula
    if cli_overrides:
        merged = _deep_merge(merged, cli_overrides)
        source += " + cli"

    # Dict'i Config objesine dönüştür
    _config = _dict_to_config(merged)
    _config._source = source

    return _config


def get_config() -> Config:
    """
    Mevcut yapılandırmayı döndürür.
    Henüz yüklenmemişse varsayılanlarla yükler.
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config() -> None:
    """Config'i sıfırla (test amaçlı)."""
    global _config
    _config = None
