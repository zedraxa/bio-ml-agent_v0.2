import os
import json
import pytest
import numpy as np
from pathlib import Path
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

# Ensure utils module is imported properly
import sys
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.model_compare import ModelComparator, compare_models

@pytest.fixture
def clf_data():
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)

@pytest.fixture
def reg_data():
    X, y = make_regression(n_samples=50, n_features=4, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def test_compare_classification(clf_data):
    """Test ModelComparator with classification task."""
    X_train, X_test, y_train, y_test = clf_data
    comparator = ModelComparator(task_type="classification", cv_folds=2)
    
    results = comparator.run(X_train, X_test, y_train, y_test, run_cv=True)
    
    assert len(results) > 0, "No classification models run"
    assert comparator.best_model_name is not None
    assert comparator._primary_metric == "accuracy"
    
    for res in results:
        assert "accuracy" in res.metrics
        assert "precision" in res.metrics
        assert "recall" in res.metrics
        assert "f1" in res.metrics

def test_compare_regression(reg_data):
    """Test ModelComparator with regression task."""
    X_train, X_test, y_train, y_test = reg_data
    comparator = ModelComparator(task_type="regression", cv_folds=2)
    
    results = comparator.run(X_train, X_test, y_train, y_test, run_cv=True)
    
    assert len(results) > 0, "No regression models run"
    assert comparator.best_model_name is not None
    assert comparator._primary_metric == "r2"
    
    for res in results:
        assert "r2" in res.metrics
        assert "mae" in res.metrics
        assert "mse" in res.metrics
        assert "rmse" in res.metrics

def test_output_json(clf_data, tmp_path):
    """Test that model comparison results export correctly to JSON format."""
    X_train, X_test, y_train, y_test = clf_data
    comparator = ModelComparator(task_type="classification", cv_folds=2)
    comparator.run(X_train, X_test, y_train, y_test, run_cv=False)
    
    out_dir = tmp_path / "results"
    saved_files = comparator.save_results(str(out_dir), save_model=False)
    
    assert "json" in saved_files
    json_path = saved_files["json"]
    
    assert json_path.exists()
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert "task_type" in data
    assert data["task_type"] == "classification"
    assert "best_model" in data
    assert "ranking" in data
    assert len(data["ranking"]) == len(comparator.results)
    
    # Check that output dict format matches expectations
    first_rank = data["ranking"][0]
    assert "model" in first_rank
    assert "metrics" in first_rank
