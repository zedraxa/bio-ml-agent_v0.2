import pytest

def test_ml_package_exports():
    """Test that all unified ML modules are correctly exported."""
    try:
        from bio_ml_agent.ml import (
            compare_models, ModelComparator,
            save_model, load_model,
            evaluate_model,
            DataPreprocessor,
            optimize_model,
            XAIEngine,
            MedicalCNN,
            ProteinAnalyzer, GenomicAnalyzer
        )
    except ImportError as e:
        pytest.fail(f"Could not import unified ml components: {e}")

    # Just asserting that they are not None 
    # (since ml/__init__.py uses try/except gracefully degrading on missing dependencies)
    assert compare_models is not None
    assert ModelComparator is not None
    assert DataPreprocessor is not None
    assert optimize_model is not None
    assert XAIEngine is not None
    assert MedicalCNN is not None
    assert ProteinAnalyzer is not None
    assert GenomicAnalyzer is not None

