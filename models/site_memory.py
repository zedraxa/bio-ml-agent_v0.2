from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# --- GÖREV 1: Site Profile Registry ---

class KnownSelectors(BaseModel):
    """Domain için bilinen sabit yollar."""
    login_button: Optional[str] = None
    username_input: Optional[str] = None
    password_input: Optional[str] = None
    pagination_next: Optional[str] = None
    accept_cookies_btn: Optional[str] = None

class SiteProfile(BaseModel):
    """AJAN'ın her bir domain için öğrendiği Site Registry Profili."""
    domain: str = Field(..., description="Örn: github.com")
    known_selectors: KnownSelectors = Field(default_factory=KnownSelectors)
    login_route: Optional[str] = Field(None, description="Direct URL of the login page")
    requires_js: bool = Field(default=True, description="Site JS olmadan çalışmıyorsa True")
    export_mechanism_detected: bool = Field(default=False)

# --- GÖREV 2: Table/Document Extraction ---

class ExtractionFormat(str, Enum):
    CSV = "csv"
    JSON = "json"
    PDF = "pdf"
    HTML = "html"
    CHART_DATA = "chart_data"

class ExtractionConfig(BaseModel):
    """Bir tablodan veya belgeden veri çıkarma ayarları."""
    target_selector: str = Field(..., description="Tablonun element CSS yolu vs.")
    desired_format: ExtractionFormat = Field(default=ExtractionFormat.CSV)
    paginate_if_possible: bool = Field(default=False, description="Sonraki sayfalara geçip tüm datayı toplayayım mı?")
    max_items: Optional[int] = Field(None, description="100 satırla sınırla vb.")

class ExtractionResult(BaseModel):
    """Veri kazıma (Extraction) sonrasında üretilen sonuç."""
    status: str = Field(..., description="'success', 'partial', 'failed'")
    extracted_rows: int = Field(default=0)
    output_workspace_path: str = Field(..., description="Nereye kaydedildi?")
    message: str = Field(default="")

# --- GÖREV 3: Browser Memory ---

class SelectorMemory(BaseModel):
    """Hangi selector ne kadar başarılı oldu?"""
    selector_str: str
    success_count: int = Field(default=0)
    fail_count: int = Field(default=0)
    last_working_at: Optional[str] = Field(None, description="ISO Date")
    
    @property
    def reliability_score(self) -> float:
        total = self.success_count + self.fail_count
        return self.success_count / total if total > 0 else 0.0

class SiteInteractionHistory(BaseModel):
    """Hangi sitede hangi akışın arşivi."""
    domain: str
    selectors_stats: List[SelectorMemory] = Field(default_factory=list)
    failed_steps_archive: List[Dict[str, Any]] = Field(default_factory=list, description="Screenshot ID ve loglar")

# --- GÖREV 4: Site Policy Hints ---

class SitePolicyHint(BaseModel):
    """Otomasyon engellerine karşı 'robots.txt' veya 'Safe Mode' kuralları."""
    domain: str
    slow_mode_enabled: bool = Field(default=False, description="Sürekli 429 (Rate Limit) dönüyorsa yavaşla")
    delay_between_clicks_ms: int = Field(default=500)
    respects_robots_txt: bool = Field(default=True)
    banned_paths: List[str] = Field(default_factory=list, description="Asla tıklama, bal küpü (honeypot)")
