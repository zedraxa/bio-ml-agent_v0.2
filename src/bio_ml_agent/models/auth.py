from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# --- GÖREV 1: Browser Profile Manager ---

class BrowserProfile(BaseModel):
    """Tarayıcının profil, çerez ve ayar şeması."""
    profile_id: str = Field(..., description="Proje veya kullanıcı bazlı benzersiz ID")
    cookie_vault_path: Optional[str] = Field(None, description="Şifreli çerezlerin yolu")
    local_storage_path: Optional[str] = None
    proxy_url: Optional[str] = None
    user_agent_override: Optional[str] = None

# --- GÖREV 2: Login Orchestration ---

class LoginFlowType(str, Enum):
    FORM_BASED = "form_based"         # Kullanıcı adı / şifre kutucukları (Klasik)
    OAUTH = "oauth"                   # Google, Github, Apple ile giriş
    MAGIC_LINK = "magic_link"         # Maile gelen benzersiz link
    SSO_REDIRECT = "sso_redirect"     # Kurumsal yönlendirme loginler
    UNKNOWN = "unknown"

class LoginState(BaseModel):
    """Bulunulan sayfanın login akış durumu."""
    flow_type: LoginFlowType = Field(default=LoginFlowType.UNKNOWN)
    is_logged_in: bool = Field(default=False)
    target_url: Optional[str] = Field(None, description="Giriş yapıldıktan sonra hedeflenen sayfa")

# --- GÖREV 3: Challenge Detector ---

class ChallengeType(str, Enum):
    CAPTCHA = "captcha"                     # ReCaptcha, hCaptcha vb. eklentiler
    TWO_FACTOR_AUTH = "two_factor_auth"     # SMS, Mail, Auth App şifresi bekleme
    RATE_LIMIT = "rate_limit"               # Cloudflare Block, 429 Too Many Requests
    SUSPICIOUS_LOGIN = "suspicious_login"     # "Yeni cihaz şüpheli giriş" uyarısı
    NONE = "none"

class ChallengeState(BaseModel):
    """Sayfaya girişte karşılaşılan engel durumu."""
    type: ChallengeType = Field(default=ChallengeType.NONE)
    is_blocking: bool = Field(default=False, description="Ajanın kendi yeteneğiyle aşamayacağı kadar sert mi?")
    detection_confidence: float = Field(default=0.0)
    challenge_metadata: Dict[str, Any] = Field(default_factory=dict, description="İmaj kaynağı, site key vs.")

# --- GÖREV 4: Human Handoff (HITL) ---

class HandoffStatus(str, Enum):
    NOT_REQUESTED = "not_requested" # Sorun yok
    PENDING_USER = "pending_user"   # Ajan işlemi durdurdu, insandan hareket bekliyor
    RESOLVED = "resolved"           # İnsan sorunu çözüp yetkiyi ajana geri verdi
    FAILED = "failed"               # İnsan çözemedi / Timeout

class HumanHandoffRequest(BaseModel):
    """Ajanın aşamadığı sorunlarda (ör: fiziksel OTP) süreci insana devretme olayı."""
    handoff_id: str
    run_id: str
    status: HandoffStatus = Field(default=HandoffStatus.NOT_REQUESTED)
    reason: str = Field(..., description="Kullanıcıya gösterilecek 'Neden durdum?' metni")
    challenge: Optional[ChallengeState] = None
    screenshot_url: Optional[str] = None
    resume_checkpoint_id: Optional[str] = None # Durdurulan işlemin kayıt ID'si (Kaldığı yer)

# --- GÖREV 5: Credential Scope Policy ---

class ClearanceLevel(str, Enum):
    LOW = "low"     # Uç hesaplar, önemsiz şifreler
    MEDIUM = "medium"
    HIGH = "high"   # Kredi kartı, AWS key vb kritik sırlar

class CredentialPolicy(BaseModel):
    """Ajanın (veya alt ajanın) hangi şifreyi nerede kullanabileceğinin kontratı."""
    policy_id: str
    credential_id: str = Field(..., description="Vault/Kasada bulunan sırrın referansı")
    allowed_domains: List[str] = Field(..., description="Hangi web sitelerine bu şifre gömülebilir? (ör: *.kaggle.com)")
    clearance_level: ClearanceLevel = Field(default=ClearanceLevel.LOW)
    max_usage_count: Optional[int] = Field(None, description="Bir günde maksimum kullanım sınırı")
