from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# --- GÖREV 1: Dosya Format Desteği ---
class BioFormat(str, Enum):
    FASTA = "fasta"
    FASTQ = "fastq"
    BAM = "bam"
    SAM = "sam"
    VCF = "vcf"
    GFF = "gff"
    GTF = "gtf"
    OMICS_CSV = "omics_csv"

class BioSequenceMeta(BaseModel):
    """Biyolojik dizi (DNA/RNA/Protein) metadatası."""
    artifact_id: str
    format: BioFormat
    organism: Optional[str] = None
    sequence_type: str # Örn: "DNA", "RNA", "Protein"
    read_count: Optional[int] = None
    avg_read_length: Optional[float] = None
    reference_genome: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

# --- GÖREV 2: Medikal Görüntü Desteği ---
class Modality(str, Enum):
    CT = "CT"
    MRI = "MRI"
    PET = "PET"
    XRAY = "XRAY"
    US = "US"
    HISTO = "HISTO" # Histopathology

class MedicalImageMeta(BaseModel):
    """Medikal görüntü (DICOM, NIfTI) metadatası."""
    artifact_id: str
    modality: Modality
    study_instance_uid: Optional[str] = None
    series_instance_uid: Optional[str] = None
    sop_instance_uid: Optional[str] = None
    body_part: Optional[str] = None
    pixel_spacing: List[float] = Field(default_factory=list) # [x, y]
    slice_thickness: Optional[float] = None
    orientation: List[float] = Field(default_factory=list) # Image Orientation (Patient)
    dimensions: List[int] = Field(default_factory=list) # [width, height, slices]

# --- GÖREV 3: QC Pipeline ---
class QCCheckType(str, Enum):
    CORRUPTION = "corruption"
    MISSING_METADATA = "missing_metadata"
    SKEWNESS = "skewness"
    INCONSISTENT_SPACING = "inconsistent_spacing"
    LOW_QUALITY_READS = "low_quality_reads"

class QCCheckResult(BaseModel):
    """Kalite kontrol kontrol sonucu."""
    check_id: str
    artifact_id: str
    check_type: QCCheckType
    is_passed: bool
    score: float = Field(ge=0.0, le=1.0)
    findings: List[str] = Field(default_factory=list)
    timestamp: str

# --- GÖREV 4: De-identification ---
class DeIdAction(str, Enum):
    REMOVE = "remove"
    MASK = "mask"
    PSEUDONYMIZE = "pseudonymize"
    KEEP = "keep"

class DeIdPolicy(BaseModel):
    """Anonimleştirme poliçesi."""
    policy_id: str
    name: str
    tag_actions: Dict[str, DeIdAction] = Field(default_factory=dict) # DICOM Tag -> Action
    remove_private_tags: bool = Field(default=True)
    anonymize_filenames: bool = Field(default=True)

class AnonymizationAudit(BaseModel):
    """Anonimleştirme denetim kaydı."""
    audit_id: str
    original_artifact_id: str
    anonymized_artifact_id: str
    policy_id: str
    agent_id: str
    timestamp: str
