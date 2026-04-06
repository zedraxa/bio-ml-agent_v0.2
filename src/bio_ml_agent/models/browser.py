from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# --- GÖREV 1: Çift Katmanlı Perception ---

class DOMElement(BaseModel):
    """DOM ve Vision katmanlarının birleştirilmiş obje modeli."""
    bio_id: str = Field(..., description="Elemente atanmış benzersiz bio-id")
    tag: str = Field(..., description="HTML Tag (BUTTON, A, INPUT)")
    role: Optional[str] = None
    label: str = Field(default="")

    # Bounding Box (Vision ve Tıklanabilirlik için)
    x: float = Field(default=0.0)
    y: float = Field(default=0.0)
    width: float = Field(default=0.0)
    height: float = Field(default=0.0)

    # Vision Confidence (Eğer resim üzerinden teyit edildiyse 1.0)
    vision_score: float = Field(default=0.0)
    is_visible: bool = Field(default=True)

class PerceptionState(BaseModel):
    """Tarayıcının anlık bütünleşik algı haritası."""
    url: str
    title: str
    interactive_elements: List[DOMElement] = Field(default_factory=list)
    screenshot_path: Optional[str] = None

# --- GÖREV 2: Selector Güvenilirlik Puanı ---

class SelectorType(str, Enum):
    CSS = "css"
    XPATH = "xpath"
    ARIA = "aria"
    TEXT = "text"
    STRUCTURAL = "structural"

class SelectorCandidate(BaseModel):
    """Tıklanacak hedefin bulunma yöntemleri ve güvencesi."""
    type: SelectorType
    value: str
    confidence_score: float = Field(ge=0.0, le=1.0, description="1.0 kesin bulur, 0.1 tahmini")

# --- GÖREV 3: Action Planner ---

class BrowserActionType(str, Enum):
    GOTO = "goto"
    CLICK = "click"
    FILL = "fill"
    PRESS = "press"
    SELECT = "select"
    SCROLL = "scroll"
    WAIT_FOR = "wait_for"
    EXTRACT = "extract_text"
    UPLOAD = "upload"         # Yeni Eklendi
    DRAG_DROP = "drag_drop"   # Yeni Eklendi
    DONE = "done"
    FAIL = "fail"

class BrowserAction(BaseModel):
    """Ajanın bir adımda yapmayı planladığı eylem modeli."""
    type: BrowserActionType
    target_selector: Optional[SelectorCandidate] = None
    target_bio_id: Optional[str] = None
    value: str = Field(default="")
    reason: str = Field(default="")
    is_multi_step: bool = Field(default=False, description="Bu eylemi yapıp hemen arkasından başka bir eylem yapılmalı mı? (Örn: Dosya seçip Enter'a basma)")

# --- GÖREV 4: State Delta Engine ---

class DeltaStatus(str, Enum):
    SUCCESS = "success"       # İstenilen etki gerçekleşti
    WAITING = "waiting"       # DOM hala yükleniyor, bekle
    FAILED = "failed"         # Hata kutucuğu çıktı veya DOM hiç değişmedi

class DOMDelta(BaseModel):
    """Önceki ile Sonraki DOM Snapshotları arasındaki fark (Diff)."""
    added_nodes_count: int = Field(default=0)
    removed_nodes_count: int = Field(default=0)
    modified_text_nodes: int = Field(default=0)
    visual_change_percent: float = Field(default=0.0, description="Ekran görüntüsündeki px farklılığı")
    status: DeltaStatus = Field(default=DeltaStatus.SUCCESS)

# --- GÖREV 5: Download/Upload Yöneticisi ---

class TransferDirection(str, Enum):
    UPLOAD = "upload"
    DOWNLOAD = "download"

class BrowserTransferEvent(BaseModel):
    """Dosya transfer durum nesnesi."""
    direction: TransferDirection
    file_name: str
    mime_type: Optional[str] = None
    workspace_path: str = Field(..., description="Workspace içindeki referans yolu")
    size_bytes: int = Field(default=0)

# --- GÖREV 6: Browser Session Timeline ---

class BrowserStepTimeline(BaseModel):
    """Ajanın tek bir adımındaki tüm olayların birleşik kaydı."""
    step_number: int
    action_taken: BrowserAction
    perception_before: PerceptionState
    delta_after: Optional[DOMDelta] = None

    # Gözlemlenebilirlik (Observability)
    network_requests: int = Field(default=0, description="Bu adımda atılan ağ/API istekleri")
    console_errors: List[str] = Field(default_factory=list)
    step_latency: float = Field(default=0.0)

# --- FAZ 1 SPRINT 1.2: Dinamik Formlar ve Bekleme Motoru ---

# Görev 1: Smart Wait Engine
class WaitCondition(str, Enum):
    NETWORK_IDLE = "network_idle"     # Ağ isteklerinin bitmesini bekle
    MUTATION_STOP = "mutation_stop"   # DOM değişikliklerinin durmasını bekle
    SPINNER_HIDDEN = "spinner_hidden" # Yükleniyor/Spinner ikonlarının kaybolması
    STALE_RECOVER = "stale_recover"   # DOM yenilenirse elemanı tekrar bul

class SmartWaitConfig(BaseModel):
    """Ajanın eylem öncesi/sonrası bekleme zekası."""
    condition: WaitCondition = Field(default=WaitCondition.NETWORK_IDLE)
    timeout_ms: int = Field(default=10000, description="Maksimum bekleme süresi")
    require_visual_stability: bool = Field(default=True, description="Sayfada hareketli (kayan) animasyon bitene kadar bekle")

# Görev 4 & 5: Rich Editor & Doğrulama
class InputFieldType(str, Enum):
    TEXT = "text"
    PASSWORD = "password"
    EMAIL = "email"
    PHONE = "phone"
    DATE = "date"
    MARKDOWN_EDITOR = "markdown_editor"
    CODE_EDITOR = "code_editor"
    CONTENT_EDITABLE = "content_editable"

# Görev 2: Form Inference Engine
class FormField(BaseModel):
    """Zenginleştirilmiş Form Elemanı (Girdi Formatını Anlama)"""
    bio_id: str
    inferred_type: InputFieldType = Field(default=InputFieldType.TEXT)
    inferred_label: str = Field(default="", description="Yakınındaki text veya placeholder'dan anlaşılan isim")
    is_required: bool = Field(default=False)
    validation_regex: Optional[str] = Field(default=None, description="Tarayıcıya yazılmadan önce TR telefon numarası vb doğrulama kuralı")

# Görev 3: Çok Adımlı Form Executor
class FormValidationState(BaseModel):
    """Sayfadayer alan formun o anki durumu."""
    has_validation_error: bool = Field(default=False)
    error_messages: List[str] = Field(default_factory=list, description="Ekranda çıkan 'Şifre çok kısa' yazıları")
    is_partial_saved: bool = Field(default=False, description="Formun bir kısmı sunucuya auto-save yapıldı mı?")
    current_step: int = Field(default=1, description="Çok adımlı (Wizard) formlarda nerede olduğumuz")
