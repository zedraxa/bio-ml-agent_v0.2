from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# --- GÖREV 1: Plugin Manifest Standardı ---

class PluginCategory(str, Enum):
    DATA_INTEGRATION = "data_integration" # Google Drive, OneDrive
    SCIENTIFIC_DB = "scientific_db"       # PubMed, Benchling, NCBI
    SOURCE_CONTROL = "source_control"     # GitHub, GitLab
    ML_PLATFORM = "ml_platform"           # HuggingFace, Kaggle
    CUSTOM = "custom"                     # Kullanıcı özel eklentileri

class PluginManifest(BaseModel):
    """Bir eklentinin (Plugin) kimliğini, yetkilerini ve araçlarını deklare ettiği kaynak belge."""
    plugin_id: str
    name: str = Field(..., description="Eklentinin görünen adı (Örn: 'PubMed Search')")
    version: str = Field(..., description="SemVer formatında sürüm (Örn: '1.2.0')")
    category: PluginCategory = Field(default=PluginCategory.CUSTOM)
    required_permissions: List[str] = Field(default_factory=list, description="İstediği yetki (Örn: 'read_vault')")
    tool_schemas: List[Dict[str, Any]] = Field(default_factory=list, description="Ajan'a kazandırdığı araçlar")
    resource_mounts: List[str] = Field(default_factory=list, description="Klasöre bağlanan kaynak (Örn: /mnt/pubmed)")

# --- GÖREV 2: Signed Plugin Yükleme ---

class SandboxClass(str, Enum):
    UNTRUSTED = "untrusted"       # Düşük erişim (İnternet yok, salt-okunur)
    SEMI_TRUSTED = "semi_trusted" # Kurumsal yalıtım (Proxy üzerinden geçişli)
    TRUSTED = "trusted"           # Çekirdek izinler verilebilir (Ajanla aynı yetki)

class PluginSignature(BaseModel):
    """Büyük marketlerden veya yayıncılardan imzalanan eklentilerin doğruluk verisi."""
    publisher_id: str
    signature_hash: str
    trust_score: float = Field(default=0.0, description="1.0 üzerinden güvenilirlik puanı (Örn: 0.98)")
    sandbox_class: SandboxClass = Field(default=SandboxClass.UNTRUSTED)

# --- GÖREV 3: Plugin Lifecycle ---

class PluginState(str, Enum):
    AVAILABLE = "available"     # Markette var, henüz indirilmedi
    INSTALLED = "installed"     # İndirildi ama aktif değil
    ENABLED = "enabled"         # Ajan bu eklentiyi şu an kullanabilir
    DISABLED = "disabled"       # Ajan tarafından kilitlendi
    ERROR = "error"             # Yüklenirken veya çalışırken çöktü

class PluginLifecycleState(BaseModel):
    """Bir eklentinin Ajan içindeki yaşam evresi."""
    plugin_id: str
    current_state: PluginState = Field(default=PluginState.AVAILABLE)
    installed_at: Optional[str] = Field(None, description="ISO Zaman Damgası")
    error_message: Optional[str] = None
    is_auto_updatable: bool = Field(default=False)

# --- GÖREV 5: Plugin Health Panel (Telemetri) ---

class PluginTelemetry(BaseModel):
    """Çalışan bir eklentinin performans ve çökme günlüklerini tutan metrik raporu."""
    plugin_id: str
    invocation_count: int = Field(default=0, description="Kaç kere çağrıldı?")
    error_rate: float = Field(default=0.0, description="Hata yüzdesi (Örn: 0.15 = %15)")
    avg_latency_ms: float = Field(default=0.0, description="Ortalama ne kadar sürede cevap verdi?")
    token_impact: int = Field(default=0, description="LLM'e eklediği tahmini maliyet token sayısı")
    recent_crash_logs: List[str] = Field(default_factory=list, description="Kapanıp kalırsa aldığı trace (izler)")
