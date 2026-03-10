from enum import Enum
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field

# --- GÖREV 1: MONAI Tabanlı Transforms ---
class TransformType(str, Enum):
    RESAMPLE = "resample"
    NORMALIZE = "normalize"
    CROP = "crop"
    PATCH = "patch"
    AUGMENT = "augment"
    ORIENT = "orient"

class ImageTransform(BaseModel):
    """Bireysel görüntü dönüşüm adımı."""
    type: TransformType
    params: Dict[str, Any] = Field(default_factory=dict)
    is_enabled: bool = Field(default=True)

class ImageTransformStack(BaseModel):
    """Ardışık görüntü dönüşüm zinciri."""
    stack_id: str
    name: str
    transforms: List[ImageTransform] = Field(default_factory=list)
    version: str
    created_at: str

# --- GÖREV 2: Skull Stripping / Organ ROI Hazırlığı ---
class ROIConfig(BaseModel):
    """İlgi alanı (Region of Interest) konfigürasyonu."""
    roi_id: str
    artifact_id: str
    center: List[float] # [x, y, z] in world or voxel coordinates
    size: List[float] # [dx, dy, dz]
    label: str # Örn: "brain", "liver", "tumor"
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

# --- GÖREV 4: Dataset Cards ---
class DatasetCard(BaseModel):
    """Veri setinin istatistiksel ve yapısal profili."""
    dataset_id: str
    name: str
    modalities: List[str]
    voxel_spacing_stats: Dict[str, Any] # Mean, Std, Min, Max
    class_distribution: Dict[str, int] # Label -> Count
    institution_split: Dict[str, float] # Institution -> Percentage
    total_samples: int
    created_at: str
    license: Optional[str] = None

# --- GÖREV 5: Annotation Workflow ---
class AnnotationStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REVIEWED = "reviewed"
    REJECTED = "rejected"

class AnnotationSession(BaseModel):
    """Etiketleme ve inceleme oturumu."""
    session_id: str
    project_id: str
    reviewer_id: Optional[str] = None
    status: AnnotationStatus = Field(default=AnnotationStatus.PENDING)
    samples_count: int
    created_at: str
    closed_at: Optional[str] = None

class ActiveLearningSample(BaseModel):
    """Aktif öğrenme için seçilmiş yüksek değerli örnek."""
    sample_id: str
    artifact_id: str
    uncertainty_score: float
    reason: str # Örn: "boundary_uncertainty", "outlier"
    priority: int = Field(default=1)
    is_processed: bool = Field(default=False)
