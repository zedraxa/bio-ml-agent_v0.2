from enum import Enum
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field

# --- GÖREV 1: Molecule Parsers & Graphs ---
class MoleculeFormat(str, Enum):
    SMILES = "smiles"
    SDF = "sdf"
    PDB = "pdb"
    INCHI = "inchi"

class MoleculeStructure(BaseModel):
    """Moleküler yapı ve grafik temsili verileri."""
    molecule_id: str
    name: Optional[str] = None
    format: MoleculeFormat
    content: str # SMILES dizisi veya dosya içeriği
    num_atoms: int
    num_bonds: int
    molecular_weight: Optional[float] = None
    graph_data_path: Optional[str] = None # İşlenmiş PyG verisi yolu
    metadata: Dict[str, Any] = Field(default_factory=dict)

# --- GÖREV 2: PyG Tabanlı GNN Şablonları ---
class GNNLayerType(str, Enum):
    GCN = "gcn"
    GAT = "gat"
    GIN = "gin"
    SAGE = "sage"
    MPNN = "mpnn"

class GNNModelConfig(BaseModel):
    """Grafik Sinir Ağı (GNN) model mimarisi."""
    model_id: str
    name: str
    layers: List[GNNLayerType]
    hidden_channels: int
    num_layers: int
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    use_edge_attr: bool = Field(default=False)
    target_type: str = Field(default="graph") # "graph", "node", "edge"

# --- GÖREV 3: Dataset Adapters ---
class MolecularDatasetCard(BaseModel):
    """MoleculeNet veya Lab bazlı veri seti kartı."""
    dataset_id: str
    source: str # Örn: "MoleculeNet", "In-house Lab"
    task_type: str # "classification", "regression"
    targets: List[str] # Örn: ["HIV_active", "toxicity"]
    split_strategy: str = Field(default="scaffold") # "random", "scaffold", "stratified"
    size: int

# --- GÖREV 4: Explainability ---
class MolecularExplanation(BaseModel):
    """Moleküler düzeyde model açıklanabilirliği."""
    explanation_id: str
    molecule_id: str
    model_id: str
    atom_importance: List[float] # Atomların öncelik skorları
    subgraph_importance: Dict[str, float] = Field(default_factory=dict) # Alt grafiklerin skorları
    method: str # Örn: "GNNExplainer", "Integrated Gradients"

# --- GÖREV 5: Candidate Ranking ---
class DrugCandidate(BaseModel):
    """İlaç adayı derecelendirme ve analiz verisi."""
    candidate_id: str
    molecule_id: str
    prediction_scores: Dict[str, float] # Özellik skorları (toksisite, çözünürlük vb.)
    uncertainty: float = Field(default=0.0, ge=0.0, le=1.0)
    novelty_score: float = Field(default=0.0, ge=0.0, le=1.0)
    synthetic_feasibility: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    rank: Optional[int] = None
    tags: List[str] = Field(default_factory=list)
