"""
Bio-ML Agent — Birleşik ML Modülleri

Re-exports from legacy ML package. All ML functions are accessible from
this single import point.
"""

# ── Model Karşılaştırma ──
try:
    from bio_ml_agent.legacy.ml.model_compare import compare_models, ModelComparator
except ImportError:
    compare_models = None
    ModelComparator = None

# ── Model Kaydetme / Yükleme ──
try:
    from bio_ml_agent.legacy.ml.model_loader import save_model, load_model
except ImportError:
    save_model = None
    load_model = None

# ── Model Değerlendirme ──
try:
    from bio_ml_agent.legacy.ml.evaluator import evaluate_model
except ImportError:
    evaluate_model = None

# ── Veri Ön İşleme ──
try:
    from bio_ml_agent.legacy.ml.preprocessor import DataPreprocessor
except ImportError:
    DataPreprocessor = None

# ── Hiperparametre Optimizasyonu ──
try:
    from bio_ml_agent.legacy.ml.hyperparameter import optimize_model
except ImportError:
    optimize_model = None

# ── XAI (Açıklanabilirlik) ──
try:
    from bio_ml_agent.legacy.ml.xai_engine import XAIEngine
except ImportError:
    XAIEngine = None

# ── Deep Learning ──
try:
    from bio_ml_agent.legacy.ml.deep_learning import MedicalCNN
except ImportError:
    MedicalCNN = None

# ── Biyomühendislik Araç Seti ──
try:
    from bio_ml_agent.legacy.ml.bioeng_toolkit import ProteinAnalyzer, GenomicAnalyzer
except ImportError:
    ProteinAnalyzer = None
    GenomicAnalyzer = None

__all__ = [
    "compare_models", "ModelComparator",
    "save_model", "load_model",
    "evaluate_model",
    "DataPreprocessor",
    "optimize_model",
    "XAIEngine",
    "MedicalCNN",
    "ProteinAnalyzer", "GenomicAnalyzer",
]
