import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

log = logging.getLogger("kernel.identity")

@dataclass
class AgentIdentity:
    """Ajanın kimlik ve yetenek profilini tanımlar."""
    name: str
    role: str
    capabilities: List[str]
    model: str
    version: str = "1.0.0"
    id: str = field(default_factory=lambda: "agent-" + str(time.time()))

class IdentityRegistry:
    """Kayıtlı ajanların kimliklerini yönetir."""
    def __init__(self):
        self.registry: Dict[str, AgentIdentity] = {}

    def register(self, identity: AgentIdentity):
        self.registry[identity.name] = identity
        log.info(f"🆔 Agent Registered: {identity.name} as {identity.role}")

    def get_by_role(self, role: str) -> List[AgentIdentity]:
        return [ident for ident in self.registry.values() if ident.role == role]
