# ml/__init__.py
# ═══════════════════════════════════════════════════════════
#  Birleşik ML Paketi — Faz 4
#  Tüm ML fonksiyonları tek import noktasından erişilebilir.
# ═══════════════════════════════════════════════════════════

"""
Bio-ML Agent — Birleşik ML Modülleri

Kullanım:
    from bio_ml_agent.ml import compare_models, evaluate_model, save_model, load_model
    from bio_ml_agent.ml import XAIEngine
    from bio_ml_agent.ml import ProteinAnalyzer, DNAAnalyzer
    from bio_ml_agent.ml import optimize_hyperparameters
    from bio_ml_agent.ml import quick_preprocess
"""

# ── Model Karşılaştırma ──
try:
    from bio_ml_agent.ml.model_compare import compare_models, ModelComparator
except ImportError:
    compare_models = None
    ModelComparator = None

# ── Model Kaydetme / Yükleme ──
try:
    from bio_ml_agent.ml.model_loader import save_model, load_model
except ImportError:
    save_model = None
    load_model = None

# ── Model Değerlendirme ──
try:
    from bio_ml_agent.ml.evaluator import evaluate_model
except ImportError:
    evaluate_model = None

# ── Veri Ön İşleme ──
try:
    from bio_ml_agent.ml.preprocessor import DataPreprocessor
except ImportError:
    DataPreprocessor = None

# ── Hiperparametre Optimizasyonu ──
try:
    from bio_ml_agent.ml.hyperparameter import optimize_model
except ImportError:
    optimize_model = None

# ── XAI (Açıklanabilirlik) ──
try:
    from bio_ml_agent.ml.xai_engine import XAIEngine
except ImportError:
    XAIEngine = None

# ── Deep Learning ──
try:
    from bio_ml_agent.ml.deep_learning import MedicalCNN
except ImportError:
    MedicalCNN = None

# ── Biyomühendislik Araç Seti ──
try:
    from bio_ml_agent.ml.bioeng_toolkit import ProteinAnalyzer, GenomicAnalyzer
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
