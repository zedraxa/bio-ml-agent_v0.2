from enum import Enum
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field

# --- GÖREV 1: Sequence Tokenizer Katmanı ---
class TokenizerStrategy(str, Enum):
    KMER = "kmer"
    BPE = "bpe"
    MIXED = "mixed"
    CHARACTER = "character"

class SequenceTokenizerConfig(BaseModel):
    """Genomik diziler için tokenizer konfigürasyonu."""
    tokenizer_id: str
    name: str
    strategy: TokenizerStrategy
    vocab_size: int
    kmer_size: Optional[int] = None # Sadece kmer stratejisi için
    overlap: bool = Field(default=True)
    padding_side: str = Field(default="right")
    special_tokens: Dict[str, int] = Field(default_factory=dict)
    version: str

# --- GÖREV 2 & 5: DNABERT-2 Embedding Pipeline & Provenance ---
class GenomicEmbeddingMeta(BaseModel):
    """Genomik embedding (vektörel temsil) metadatası ve menşei."""
    embedding_id: str
    artifact_id: str # Orijinal dizi artifact ID'si
    model_id: str # Örn: "dnabert2-base"
    tokenizer_id: str
    dimension: int
    pooling_strategy: str # Örn: "mean", "cls", "max"
    species: Optional[str] = None
    creation_timestamp: str
    preprocessing_signature: str # Ön işleme adımlarının hash'i
    model_version: str

# --- GÖREV 3: Sequence Tasks ---
class GenomicTaskType(str, Enum):
    PROMOTER_PREDICTION = "promoter_prediction"
    TFBS_DETECTION = "tfbs_detection"
    ENHANCER_CLASSIFICATION = "enhancer_classification"
    SPECIES_CLUSTERING = "species_clustering"
    VARIANT_EFFECT_PREDICTION = "variant_effect_prediction"

class GenomicTaskConfig(BaseModel):
    """Genetik dizi analiz görev şablonu."""
    task_id: str
    name: str
    task_type: GenomicTaskType
    required_context_length: int
    labels: List[str]
    metrics: List[str] = Field(default_factory=lambda: ["accuracy", "auroc", "auprc"])

# --- GÖREV 4: Long Sequence Handling ---
class ChunkingStrategy(str, Enum):
    FIXED_SIZE = "fixed_size"
    SLIDING_WINDOW = "sliding_window"
    OVERLAPPING = "overlapping"

class LongSequenceStrategy(BaseModel):
    """Uzun dizilerin parçalanması ve yönetimi stratejisi."""
    strategy_id: str
    chunk_size: int
    overlap_size: int
    chunking_type: ChunkingStrategy
    pooling: Optional[str] = "mean" # Parçaları birleştirme yöntemi
