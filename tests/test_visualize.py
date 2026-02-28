import pytest
import os
import numpy as np
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Ensure the project root is in sys.path
import sys
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.visualize import (
    plot_confusion_matrix,
    plot_roc_curve,
    MLVisualizer
)

@pytest.fixture
def sample_data():
    X, y = make_classification(n_samples=50, n_features=4, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

def test_confusion_matrix_png(sample_data, tmp_path):
    """Test generating and saving a confusion matrix PNG."""
    model, _, X_test, _, y_test = sample_data
    y_pred = model.predict(X_test)
    
    out_file = tmp_path / "cm.png"
    result_path = plot_confusion_matrix(y_test, y_pred, output_path=out_file)
    
    assert result_path.exists()
    assert result_path.suffix == ".png"

def test_roc_curve_png(sample_data, tmp_path):
    """Test generating and saving a ROC curve PNG."""
    model, _, X_test, _, y_test = sample_data
    
    out_file = tmp_path / "roc.png"
    result_path = plot_roc_curve(model, X_test, y_test, output_path=out_file)
    
    assert result_path.exists()
    assert result_path.suffix == ".png"

def test_output_directory_creation(sample_data, tmp_path):
    """Test if the MLVisualizer automatically creates output directories."""
    model, X_train, X_test, y_train, y_test = sample_data
    
    # Nested non-existent directory
    out_dir = tmp_path / "nested" / "plots"
    assert not out_dir.exists()
    
    viz = MLVisualizer(output_dir=str(out_dir))
    saved = viz.plot_all(model, X_train, X_test, y_train, y_test)
    
    assert out_dir.exists()
    assert "confusion_matrix" in saved
    assert saved["confusion_matrix"].exists()

def test_all_plots(sample_data, tmp_path):
    """Test plot_all generates the expected 6 plots."""
    model, X_train, X_test, y_train, y_test = sample_data
    
    out_dir = tmp_path / "all_plots"
    viz = MLVisualizer(output_dir=str(out_dir))
    
    import pandas as pd
    df = pd.DataFrame(X_train, columns=[f"f{i}" for i in range(X_train.shape[1])])
    
    saved = viz.plot_all(
        model, X_train, X_test, y_train, y_test, 
        feature_names=["f1", "f2", "f3", "f4"],
        df=df,
        task_type="classification"
    )
    
    # We expect several plots depending on capabilities:
    # 1. class_distribution
    # 2. confusion_matrix
    # 3. confusion_matrix_norm
    # 4. roc_curve
    # 5. feature_importance
    # 6. correlation_matrix
    # 7. learning_curve
    
    expected_keys = [
        "class_distribution", 
        "confusion_matrix",
        "roc_curve",
        "feature_importance", 
        "correlation_matrix", 
        "learning_curve"
    ]
    
    for key in expected_keys:
        assert key in saved
        assert saved[key].exists()
