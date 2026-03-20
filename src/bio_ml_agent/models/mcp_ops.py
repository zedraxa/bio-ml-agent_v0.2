from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# --- GÖREV 1 & 2: MCP İstemci / Sunucu ---

class MCPResourceType(str, Enum):
    LOCAL_FILE = "local_file"
    REMOTE_URL = "remote_url"
    DATABASE = "database"
    UNKNOWN = "unknown"

class MCPResource(BaseModel):
    """MCP üzerinden dışarıya açık veya dışarıdan alınan statik kaynak."""
    resource_id: str
    name: str
    description: Optional[str] = None
    type: MCPResourceType = Field(default=MCPResourceType.UNKNOWN)
    uri: str = Field(..., description="Bağlantı adresi (file://... veya https://...)")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class MCPTool(BaseModel):
    """Sistemin dışarı (başka ajanlara) açtığı eklentisi veya fonksiyonu."""
    tool_id: str
    name: str = Field(..., description="Function name used in the protocol")
    description: str
    input_schema: Dict[str, Any] = Field(..., description="JSON schema for arguments")
    requires_approval: bool = Field(default=False)

class MCPServerConfig(BaseModel):
    """Ajanın MCP üzerinden konuştuğu diğer sunucuların (ör: veritabanı eklentisi) ayarları."""
    server_name: str
    command: str = Field(..., description="Örn: npx, python, node")
    args: List[str] = Field(default_factory=list, description="Kalkarken verilecek bayraklar (Örn: -y, -m)")
    env_vars: Dict[str, str] = Field(default_factory=dict)

# --- GÖREV 3: Tool Permission Broker ---

class ToolPermissionScope(str, Enum):
    READ_ONLY = "read_only"
    WRITE_STRICT = "write_strict"           # Sadece belli klasörlere yazabilir
    EXECUTE_CONTAINER = "execute_container" # Docker içinde kod koşturabilir
    FULL_ACCESS = "full_access"

class UserConsent(BaseModel):
    """Kullanıcının kritik bir tool veya klasöre verdiği iznin kaydı."""
    consent_id: str
    tool_name: str
    granted_at: str = Field(..., description="ISO formattaki zaman damgası")
    expires_in_seconds: Optional[int] = Field(None, description="None = Sonsuz izin, Int = Sınıreli izin")
    auto_approve_similar: bool = Field(default=False, description="'Bunu bir daha sorma, hep yap' seçeneği")

class ProjectAllowlist(BaseModel):
    """Ajanın 'kesinlikle' dokunabileceği veya okuyabileceği proje bazlı yollar."""
    project_id: str
    allowed_read_paths: List[str] = Field(default_factory=list)
    allowed_write_paths: List[str] = Field(default_factory=list)
    banned_tools: List[str] = Field(default_factory=list, description="Örn: 'rm_rf' bu projede yasak.")

# --- GÖREV 4: Prompt/Resource Catalog ---

class PromptTemplateType(str, Enum):
    WORKFLOW = "workflow"                   # Adım adım karmaşık planlar (`how_to_deploy.md`)
    SYSTEM_INSTRUCTION = "system_instruction"# Karakter / Kimlik atayan sistem metinleri
    REASONING_HINT = "reasoning_hint"       # "Şunu çözmek için adımlar..." diyen akıl yürütme ipuçları

class PromptTemplatePack(BaseModel):
    """Belirli görevleri çözmek için yüklenen Prompt (Prompt) şablon paketleri."""
    pack_id: str
    label: str
    pack_type: PromptTemplateType = Field(default=PromptTemplateType.WORKFLOW)
    template_content: str = Field(..., description="Jinja veya formatlanabilir metin bloğu")
    required_variables: List[str] = Field(default_factory=list)

class DomainKnowledgePack(BaseModel):
    """Ajanın bir alana (Biyoloji, ML, DevOps) özgü harici olarak yüklediği zeka kaynağı."""
    knowledge_id: str
    domain: str = Field(..., description="Örn: 'bioinformatics'")
    attached_resources: List[str] = Field(default_factory=list, description="Katalogdan alınacak MCPResource ID'leri")
    description: str = Field(default="")
