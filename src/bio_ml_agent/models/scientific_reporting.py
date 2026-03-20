from enum import Enum
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field

# --- GÖREV 1: Structured Report Generator ---
class ReportSection(BaseModel):
    """Raporun bir bölümü (Örn: Abstract, Methods)."""
    title: str
    content: str
    subsections: List['ReportSection'] = Field(default_factory=list)

class ScientificReport(BaseModel):
    """Yapılandırılmış akademik rapor."""
    report_id: str
    title: str
    authors: List[str]
    abstract: str
    sections: List[ReportSection]
    limitations: List[str] = Field(default_factory=list)
    acknowledgments: Optional[str] = None
    creation_timestamp: str

# --- GÖREV 2: Citation Binder ---
class CitationType(str, Enum):
    DIRECT_QUOTE = "direct_quote"
    PARAPHRASE = "paraphrase"
    DATA_SOURCE = "data_source"
    METHOD_REFERENCE = "method_reference"

class CitationRecord(BaseModel):
    """Metin içi alıntı kaydı."""
    citation_id: str
    source_id: str # model/research.py ScientificSource.source_id
    text_segment: str # Alıntılanan paragraf veya cümle
    citation_type: CitationType
    is_supported: bool = Field(default=True)
    validation_note: Optional[str] = None

# --- GÖREV 3: Figure/Table Generator ---
class VisualType(str, Enum):
    ROC_CURVE = "roc_curve"
    CONFUSION_MATRIX = "confusion_matrix"
    ABLATION_TABLE = "ablation_table"
    DISTRIBUTION_PLOT = "distribution_plot"
    NETWORK_GRAPH = "network_graph"

class TechnicalFigure(BaseModel):
    """Bilimsel görsel metadatası."""
    figure_id: str
    title: str
    visual_type: VisualType
    artifact_path: str # Üretilen resim dosyasının yolu
    caption: str
    metrics_shown: List[str] = Field(default_factory=list)

class ScientificTable(BaseModel):
    """Bilimsel tablo metadatası."""
    table_id: str
    title: str
    headers: List[str]
    rows: List[List[Any]]
    caption: Optional[str] = None

# --- GÖREV 4: Repro Appendix ---
class ReproAppendix(BaseModel):
    """Yeniden üretilebilirlik (Reproducibility) eki."""
    appendix_id: str
    experiment_id: str # model/lifecycle.py ExperimentRun.run_id
    repro_bundle_id: str # model/lifecycle.py ReproBundle.bundle_id
    environment_info: Dict[str, str] # OS, Python version
    package_versions: Dict[str, str] # package -> version
    random_seed: int
    hardware_specs: Dict[str, Any] # GPU, RAM info
