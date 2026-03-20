from enum import Enum
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field

# --- GÖREV 1 & 2: Storage & Scoped Access ---
class SecretType(str, Enum):
    PASSWORD = "password"
    API_KEY = "api_key"
    SSH_KEY = "ssh_key"
    OAUTH_TOKEN = "oauth_token"
    CERTIFICATE = "certificate"

class AccessScope(BaseModel):
    """Sırrın kullanımını kısıtlayan kapsam."""
    allowed_domains: List[str] = Field(default_factory=list, description="Örn: ['*.github.com']")
    allowed_tools: List[str] = Field(default_factory=list, description="Örn: ['browser_agent']")
    allowed_roles: List[str] = Field(default_factory=list, description="Örn: ['coder']")
    requires_approval: bool = Field(default=False)

class SecretIdentifier(BaseModel):
    """Bir sırrın (secret) kimliği ve kısıtları."""
    secret_id: str
    key_name: str
    secret_type: SecretType
    scope: AccessScope = Field(default_factory=AccessScope)
    description: Optional[str] = None
    created_at: str
    last_accessed: Optional[str] = None

# --- GÖREV 3: Secret Leasing ---
class SecretLease(BaseModel):
    """Süreli sır erişimi ve kiralama politikası."""
    lease_id: str
    secret_id: str
    token_value: str
    issued_at: str
    expires_at: str
    ttl_seconds: int
    auto_revoke: bool = Field(default=True)
    refresh_count: int = Field(default=0)

# --- GÖREV 4: Artifact Redaction ---
class RedactionType(str, Enum):
    MASK = "mask"       # **** gibi kapatma
    REMOVE = "remove"   # Tamamen silme
    REPLACE = "replace" # Sahte veriyle değiştirme (PII için)

class RedactionRule(BaseModel):
    """Hassas verileri maskeleme kuralı."""
    rule_id: str
    pattern: str  # Regex veya keyword
    type: RedactionType = Field(default=RedactionType.MASK)
    replacement: Optional[str] = None
    description: str

class RedactedArtifact(BaseModel):
    """Maskelenmiş varlık (log, screenshot vb.) kaydı."""
    artifact_id: str
    original_path: str
    redacted_path: str
    applied_rules: List[str] # Rule ID listesi
    redacted_at: str

# --- GÖREV 5: Audit Receipts ---
class SecretAuditLog(BaseModel):
    """Sır erişim denetim izi."""
    audit_id: str
    secret_id: str
    agent_id: str
    task_id: str
    timestamp: str
    action: str  # read, write, revoke, lease
    status: str  # success, denied
    reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
