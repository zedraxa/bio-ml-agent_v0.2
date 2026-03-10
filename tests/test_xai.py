import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.xai_engine import XAIEngine
import os

@pytest.fixture
def sample_data():
    X = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100) * 2,
        'feature3': np.random.rand(100) - 0.5
    })
    y = np.random.randint(0, 2, 100)
    return X, y

@pytest.fixture
def trained_model(sample_data):
    X, y = sample_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model

def test_xai_initialization(sample_data, trained_model):
    X, _ = sample_data
    xai = XAIEngine(trained_model, X, feature_names=X.columns.tolist(), task_type="classification")
    assert xai.model is trained_model
    assert xai.task_type == "classification"

def test_generate_shap_summary(sample_data, trained_model, tmp_path):
    X, _ = sample_data
    xai = XAIEngine(trained_model, X, feature_names=X.columns.tolist(), task_type="classification")
    
    # Test generation
    xai.generate_shap_summary(X.head(), output_dir=str(tmp_path), max_display=2)
    
    # Check if file created
    expected_file = tmp_path / "shap_summary_plot.png"
    assert expected_file.exists()

def test_explain_instance_lime(sample_data, trained_model, tmp_path):
    X, _ = sample_data
    xai = XAIEngine(trained_model, X, feature_names=X.columns.tolist(), task_type="classification")
    
    # Get one instance
    instance = X.iloc[0]
    
    # Test generation
    xai.explain_instance_lime(instance, output_dir=str(tmp_path))
    
    # Check if file created
    expected_file = tmp_path / "lime_explanation.html"
    assert expected_file.exists()
