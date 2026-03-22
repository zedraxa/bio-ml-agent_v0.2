from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field
import time
from bio_ml_agent.llm_backend import auto_create_backend

class Confidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Evidence(BaseModel):
    source: str
    content_snippet: str
    timestamp: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AgentResult(BaseModel):
    success: bool
    data: Any
    confidence: Confidence
    evidence: List[Evidence] = Field(default_factory=list)
    artifacts: List[str] = Field(default_factory=list) # Yerel dosya yolları
    message: str

class BaseSubAgent(ABC):
    """
    Bio-ML Agent ekosistemindeki tüm uzman ajanlar için temel sınıf.
    'perceive -> plan -> act -> verify -> summarize' kontratını uygular.
    """
    
    def __init__(self, name: str = "", model_name: str = "gemini-2.5-flash", **kwargs):
        self.name = name or kwargs.get("role_name", self.__class__.__name__)
        self.model_name = model_name
        self.system_prompt = kwargs.get("system_prompt", "")
        self.llm = auto_create_backend(self.model_name)
        self.history = []

    @abstractmethod
    def perceive(self, context: Dict[str, Any]) -> None:
        """Ortamı veya girdiyi anlama aşaması."""
        raise NotImplementedError

    @abstractmethod
    def plan(self, goal: str) -> List[str]:
        """Hedefi adımlara bölme aşaması."""
        raise NotImplementedError

    @abstractmethod
    def act(self, step: str) -> Any:
        """Tek bir adımı yürütme aşaması."""
        raise NotImplementedError

    @abstractmethod
    def verify(self, action_result: Any) -> bool:
        """Eylemin istenen etkiyi yaratıp yaratmadığını doğrulama aşaması."""
        raise NotImplementedError

    @abstractmethod
    def summarize(self) -> AgentResult:
        """Görev sonuçlarını birleştirme ve raporlama aşaması."""
        raise NotImplementedError
