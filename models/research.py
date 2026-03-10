from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# --- GÖREV 1: Literature Search Adapters & DOI ---
class SourcePlatform(str, Enum):
    PUBMED = "pubmed"
    CROSSREF = "crossref"
    BIORXIV = "biorxiv"
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    LOCAL_PDF = "local_pdf"

class ScientificSource(BaseModel):
    """Münferit bir bilimsel kaynak (makale, preprint vb.)."""
    source_id: str
    title: str
    authors: List[str]
    platform: SourcePlatform
    doi: Optional[str] = None
    url: Optional[str] = None
    abstract: Optional[str] = None
    publication_date: Optional[str] = None
    is_peer_reviewed: bool = Field(default=False)
    citation_count: int = Field(default=0)
    keywords: List[str] = Field(default_factory=list)

# --- GÖREV 2: Source Quality Scorer ---
class SourceType(str, Enum):
    PEER_REVIEWED = "peer_reviewed"
    PREPRINT = "preprint"
    DATASET_PAPER = "dataset_paper"
    GREY_LITERATURE = "grey_literature"
    BLOG_TUTORIAL = "blog_tutorial"

class SourceQualityScore(BaseModel):
    """Kaynağın bilimsel güvenilirlik skoru."""
    source_id: str
    source_type: SourceType
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    factors: Dict[str, float] = Field(default_factory=dict) # journal_impact, citations, etc.
    review_status: str # "verified", "pending", "unverified"

# --- GÖREV 3: Evidence Graph ---
class EvidenceType(str, Enum):
    SUPPORTING = "supporting"
    CONTRADICTING = "contradicting"
    NEUTRAL = "neutral"
    METHODOLOGICAL = "methodological"

class EvidenceNode(BaseModel):
    """İddia ve kaynak arasındaki kanıt bağı."""
    source_id: str
    evidence_type: EvidenceType
    snippet: str # Kanıt teşkil eden metin parçası
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    page_number: Optional[int] = None

class ResearchClaim(BaseModel):
    """Bilimsel bir iddia ve onu destekleyen kanıtlar."""
    claim_id: str
    claim_text: str
    evidence_nodes: List[EvidenceNode] = Field(default_factory=list)
    consensus_score: float = Field(default=0.0) # Kanıtlara göre genel fikir birliği
    status: str = Field(default="under_review") # "validated", "debunked", "under_review"

# --- GÖREV 4: Contradiction Detector ---
class ContradictionReport(BaseModel):
    """İki veya daha fazla kaynak arasındaki çelişki raporu."""
    report_id: str
    claim_id: str
    contradicting_source_ids: List[str]
    details: str
    severity: str # "high", "medium", "low"
    timestamp: str
