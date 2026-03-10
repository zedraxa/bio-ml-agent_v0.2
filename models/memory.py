from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# --- GÖREV 1: Merkezi Bellek Kapsamı ---
class MemoryScope(str, Enum):
    RUN = "run"
    PROJECT = "project"
    DOMAIN = "domain"
    USER_PREFERENCE = "user_preference"

# --- GÖREV 2: Etiketli Hafıza Katmanları ---
class MemoryLayer(str, Enum):
    BIO = "bio-layer"
    ML = "ml-layer"
    BROWSER = "browser-layer"
    REVIEW = "review-layer"
    DECISION = "decision-layer"

# --- GÖREV 3: Hafıza Güven Puanı ---
class TrustLevel(str, Enum):
    OBSERVED = "observed"
    INFERRED = "inferred"
    USER_CONFIRMED = "user-confirmed"
    EXTERNAL_SOURCE_BACKED = "external-source-backed"

class MemoryTrustScore(BaseModel):
    level: TrustLevel
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    reason: Optional[str] = None

class MemoryEntry(BaseModel):
    """Bellek girişi: Katman, kapsam ve güven puanı ile birlikte."""
    entry_id: str
    content: Any
    scope: MemoryScope = Field(default=MemoryScope.PROJECT)
    layer: MemoryLayer
    tags: List[str] = Field(default_factory=list)
    trust_score: MemoryTrustScore
    created_by: str = Field(..., description="Ajan ID veya Sistem")
    created_at: str = Field(..., description="ISO Zaman Damgası")
    metadata: Dict[str, Any] = Field(default_factory=dict)

# --- GÖREV 4: Conflict Resolution ---
class ConflictStrategy(str, Enum):
    LATEST_WINS = "latest_wins"
    REVIEW_REQUIRED = "review_required"
    MERGE = "merge"

class MemoryConflict(BaseModel):
    """Çelişkili bellek girişlerini takip eden yapı."""
    conflict_id: str
    key: str
    entries: List[MemoryEntry]
    strategy: ConflictStrategy = Field(default=ConflictStrategy.REVIEW_REQUIRED)
    status: str = Field(default="pending") # pending, resolved

# --- GÖREV 5: Multitenancy ---
class MemoryTenantMeta(BaseModel):
    """Çoklu proje/tenant izolasyonu için meta veri."""
    project_id: str
    tenant_id: str
    access_level: str = Field(default="read_write")
