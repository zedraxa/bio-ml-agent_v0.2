from enum import Enum
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field

# --- GÖREV 1 & 2: nnU-Net ve MONAI Model Zoo ---
class ModelFramework(str, Enum):
    NNUNET = "nnunet"
    MONAI = "monai"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"

class BioMedicalModelConfig(BaseModel):
    """Biyomedikal model konfigürasyonu (nnU-Net, MONAI vb.)."""
    model_id: str
    name: str
    framework: ModelFramework
    version: str
    weights_url: Optional[str] = None
    architecture: str # Örn: "SegResNet", "SwinUNETR", "3d_fullres"
    input_shape: List[int] # [channels, x, y, z]
    output_classes: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

# --- GÖREV 3 & 4: Segmentation Tasks ---
class SegmentationTarget(str, Enum):
    ORGAN = "organ"
    LESION = "lesion"
    TUMOR = "tumor"
    VESSEL = "vessel"
    BONE = "bone"

class SegmentationTask(BaseModel):
    """Spesifik bir segmentasyon görevi şablonu."""
    task_id: str
    name: str
    target: SegmentationTarget
    target_name: str # Örn: "Liver", "Glioblastoma"
    base_model_id: str
    required_modality: List[str] # ["MRI", "CT"]
    is_3d: bool = Field(default=True)

# --- GÖREV 5: Inference Tiling ve Postprocessing ---
class InferenceStrategy(str, Enum):
    SLIDING_WINDOW = "sliding_window"
    FULL_IMAGE = "full_image"
    TILING = "tiling"

class PostProcessingAction(str, Enum):
    KEEP_LARGEST_COMPONENT = "keep_largest_component"
    FILL_HOLES = "fill_holes"
    GAUSSIAN_SMOOTHING = "gaussian_smoothing"
    THRESHOLD = "threshold"

class InferenceConfig(BaseModel):
    """Çıkarım (Inference) stratejisi ve parametreleri."""
    config_id: str
    strategy: InferenceStrategy = Field(default=InferenceStrategy.SLIDING_WINDOW)
    overlap: float = Field(default=0.25, ge=0.0, le=1.0)
    tile_size: List[int] = Field(default_factory=lambda: [96, 96, 96])
    post_processing: List[PostProcessingAction] = Field(default_factory=list)

# --- GÖREV 6: Uncertainty Maps ve Failure Cases ---
class UncertaintyMethod(str, Enum):
    MONTE_CARLO_DROPOUT = "mc_dropout"
    ENSEMBLE = "ensemble"
    TEST_TIME_AUGMENTATION = "tta"
    SOFTMAX_ENTROPY = "entropy"

class SegmentationReport(BaseModel):
    """Segmentasyon sonuç ve belirsizlik raporu."""
    report_id: str
    artifact_id: str
    task_id: str
    model_id: str
    dice_score: Optional[float] = None
    uncertainty_method: Optional[UncertaintyMethod] = None
    mean_uncertainty: Optional[float] = None
    failure_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    findings: List[str] = Field(default_factory=list)
    timestamp: str
