from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# --- GÖREV 1: Debate Loop & Roles ---
class DebateRole(str, Enum):
    PROPOSER = "proposer"
    CRITIC = "critic"
    VERIFIER = "verifier"
    JUDGE = "judge"

class DebateEntry(BaseModel):
    agent_id: str
    role: DebateRole
    content: str
    timestamp: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DebateSession(BaseModel):
    """Ajanlar arası tartışma oturumu."""
    session_id: str
    topic: str
    entries: List[DebateEntry] = Field(default_factory=list)
    final_judgement: Optional[str] = None
    is_closed: bool = Field(default=False)

# --- GÖREV 2: Peer Review Agent ---
class ReviewOpinion(BaseModel):
    """Kod veya araştırma inceleme sonuçları."""
    reviewer_id: str
    syntax_score: float = Field(default=1.0, ge=0.0, le=1.0)
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    logic_score: float = Field(default=1.0, ge=0.0, le=1.0)
    missing_tests: List[str] = Field(default_factory=list)
    security_issues: List[str] = Field(default_factory=list)
    comment: str

# --- GÖREV 3 & 4: Consensus & Deadlock Resolution ---
class ConsensusStatus(str, Enum):
    ONGOING = "ongoing"
    REACHED = "reached"
    STALEMATE = "stalemate" # Deadlock durumu
    HUMAN_INTERVENTION_REQUIRED = "human_intervention_required"

class ConsensusState(BaseModel):
    """Fikir birliği durumunu temsil eden yapı."""
    task_id: str
    status: ConsensusStatus = Field(default=ConsensusStatus.ONGOING)
    agreed_result: Optional[Any] = None
    dissenting_opinions: List[Dict[str, str]] = Field(default_factory=list)

class ResolutionAction(BaseModel):
    """Deadlock durumunda alınacak aksiyon."""
    action_type: str # abort, human_review, force_vote
    reason: str
    initiated_by: str

# --- GÖREV 5: Final Answer Synthesis ---
class SynthesisReport(BaseModel):
    """Farklı ajan bulgularının sentezlenmiş hali."""
    report_id: str
    contributing_agents: List[str]
    summary: str
    key_findings: List[str]
    conflicts_resolved: List[str] = Field(default_factory=list)
    final_output: Any
    created_at: str
