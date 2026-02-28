# tests/test_preprocessor.py
# ═══════════════════════════════════════════════════════════
#  Bio-ML Agent — Data Preprocessor Test Suite
# ═══════════════════════════════════════════════════════════

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.preprocessor import (
    DataPreprocessor,
    analyze_data_quality,
    detect_outliers_iqr,
    detect_outliers_zscore,
    quick_preprocess,
)


# ─────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def sample_X():
    """100 satır, 5 sütun temiz veri."""
    np.random.seed(42)
    return np.random.randn(100, 5)


@pytest.fixture
def sample_y():
    np.random.seed(42)
    return np.random.randint(0, 2, 100)


@pytest.fixture
def nan_X():
    """NaN değerler içeren veri."""
    np.random.seed(42)
    X = np.random.randn(50, 4)
    X[0, 0] = np.nan
    X[5, 2] = np.nan
    X[10, 1] = np.nan
    X[20, 3] = np.nan
    return X


@pytest.fixture
def outlier_X():
    """Outlier'lar içeren veri."""
    np.random.seed(42)
    X = np.random.randn(100, 3)
    X[0] = [100, 200, 300]  # Aşırı outlier
    X[1] = [-100, -200, -300]
    return X


# ─────────────────────────────────────────────
#  Outlier Detection Tests
# ─────────────────────────────────────────────

class TestOutlierDetection:

    def test_iqr_detects_outliers(self, outlier_X):
        mask = detect_outliers_iqr(outlier_X)
        assert mask[0] == True  # İlk satır outlier
        assert mask[1] == True

    def test_iqr_returns_boolean_array(self, sample_X):
        mask = detect_outliers_iqr(sample_X)
        assert mask.dtype == bool
        assert len(mask) == len(sample_X)

    def test_zscore_detects_outliers(self, outlier_X):
        mask = detect_outliers_zscore(outlier_X)
        assert mask[0] == True
        assert mask[1] == True

    def test_zscore_threshold(self, sample_X):
        mask_strict = detect_outliers_zscore(sample_X, threshold=2.0)
        mask_relaxed = detect_outliers_zscore(sample_X, threshold=4.0)
        assert mask_strict.sum() >= mask_relaxed.sum()


# ─────────────────────────────────────────────
#  DataPreprocessor Init Tests
# ─────────────────────────────────────────────

class TestPreprocessorInit:

    def test_default_init(self):
        pp = DataPreprocessor()
        assert pp.impute_strategy == "median"
        assert pp.scale_method == "standard"
        assert pp._is_fitted is False

    def test_custom_init(self):
        pp = DataPreprocessor(
            impute_strategy="mean",
            scale_method="minmax",
            detect_outliers="iqr",
            poly_degree=2,
        )
        assert pp.impute_strategy == "mean"
        assert pp.scale_method == "minmax"
        assert pp.detect_outliers == "iqr"
        assert pp.poly_degree == 2


# ─────────────────────────────────────────────
#  Fit & Transform Tests
# ─────────────────────────────────────────────

class TestFitTransform:

    def test_fit_returns_self(self, sample_X):
        pp = DataPreprocessor()
        result = pp.fit(sample_X)
        assert result is pp

    def test_fit_sets_fitted(self, sample_X):
        pp = DataPreprocessor()
        pp.fit(sample_X)
        assert pp._is_fitted is True

    def test_transform_before_fit_raises(self, sample_X):
        pp = DataPreprocessor()
        with pytest.raises(RuntimeError, match="fit"):
            pp.transform(sample_X)

    def test_fit_transform_returns_array(self, sample_X):
        pp = DataPreprocessor()
        result = pp.fit_transform(sample_X)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == sample_X.shape[0]

    def test_standard_scaling(self, sample_X):
        pp = DataPreprocessor(scale_method="standard")
        X_out = pp.fit_transform(sample_X)
        # Ölçeklenmiş verinin ortalaması ~0, std ~1
        assert abs(X_out.mean()) < 0.5
        assert abs(X_out.std() - 1.0) < 0.5

    def test_minmax_scaling(self, sample_X):
        pp = DataPreprocessor(scale_method="minmax")
        X_out = pp.fit_transform(sample_X)
        assert X_out.min() >= -0.01
        assert X_out.max() <= 1.01

    def test_no_scaling(self, sample_X):
        pp = DataPreprocessor(scale_method=None)
        X_out = pp.fit_transform(sample_X)
        np.testing.assert_array_almost_equal(X_out, sample_X)


# ─────────────────────────────────────────────
#  Imputation Tests
# ─────────────────────────────────────────────

class TestImputation:

    def test_nan_filled(self, nan_X):
        pp = DataPreprocessor(impute_strategy="median", scale_method=None)
        X_out = pp.fit_transform(nan_X)
        assert not np.isnan(X_out).any()

    def test_mean_impute(self, nan_X):
        pp = DataPreprocessor(impute_strategy="mean", scale_method=None)
        X_out = pp.fit_transform(nan_X)
        assert not np.isnan(X_out).any()

    def test_no_nan_no_imputer(self, sample_X):
        pp = DataPreprocessor(scale_method=None)
        pp.fit(sample_X)
        assert pp._imputer is None


# ─────────────────────────────────────────────
#  Outlier Removal Tests
# ─────────────────────────────────────────────

class TestOutlierRemoval:

    def test_outlier_removal_reduces_rows(self, outlier_X, sample_y):
        y = np.random.randint(0, 2, len(outlier_X))
        pp = DataPreprocessor(
            detect_outliers="iqr",
            remove_outliers=True,
            scale_method=None,
        )
        X_out, y_out = pp.fit_transform(outlier_X, y)
        assert X_out.shape[0] < outlier_X.shape[0]
        assert len(y_out) == X_out.shape[0]

    def test_no_removal_keeps_rows(self, outlier_X):
        pp = DataPreprocessor(
            detect_outliers="iqr",
            remove_outliers=False,
            scale_method=None,
        )
        X_out = pp.fit_transform(outlier_X)
        assert X_out.shape[0] == outlier_X.shape[0]


# ─────────────────────────────────────────────
#  PCA Tests
# ─────────────────────────────────────────────

class TestPCA:

    def test_pca_reduces_dimensions(self, sample_X):
        pp = DataPreprocessor(pca_components=2)
        X_out = pp.fit_transform(sample_X)
        assert X_out.shape[1] == 2

    def test_pca_variance_ratio(self, sample_X):
        pp = DataPreprocessor(pca_components=3)
        pp.fit_transform(sample_X)
        s = pp.summary()
        assert "pca_explained_variance" in s
        assert 0 < s["pca_explained_variance"] <= 1.0


# ─────────────────────────────────────────────
#  Polynomial Features Tests
# ─────────────────────────────────────────────

class TestPolynomial:

    def test_poly_increases_features(self, sample_X):
        pp = DataPreprocessor(poly_degree=2)
        X_out = pp.fit_transform(sample_X)
        assert X_out.shape[1] > sample_X.shape[1]

    def test_poly_interaction_only(self, sample_X):
        pp1 = DataPreprocessor(poly_degree=2, poly_interaction_only=False)
        pp2 = DataPreprocessor(poly_degree=2, poly_interaction_only=True)
        X1 = pp1.fit_transform(sample_X)
        X2 = pp2.fit_transform(sample_X)
        assert X2.shape[1] < X1.shape[1]


# ─────────────────────────────────────────────
#  Summary Tests
# ─────────────────────────────────────────────

class TestSummary:

    def test_summary_keys(self, sample_X):
        pp = DataPreprocessor()
        pp.fit_transform(sample_X)
        s = pp.summary()
        assert "is_fitted" in s
        assert "impute_strategy" in s
        assert "original_shape" in s

    def test_summary_text(self, sample_X):
        pp = DataPreprocessor(pca_components=2)
        pp.fit_transform(sample_X)
        text = pp.summary_text()
        assert "Veri Ön İşleme" in text
        assert "PCA" in text


# ─────────────────────────────────────────────
#  Utility Functions Tests
# ─────────────────────────────────────────────

class TestQuickPreprocess:

    def test_basic(self, sample_X):
        X_out = quick_preprocess(sample_X)
        assert isinstance(X_out, np.ndarray)
        assert X_out.shape[0] == sample_X.shape[0]

    def test_with_pca(self, sample_X):
        X_out = quick_preprocess(sample_X, pca=2)
        assert X_out.shape[1] == 2

    def test_with_outlier_removal(self, outlier_X):
        y = np.random.randint(0, 2, len(outlier_X))
        X_out, y_out = quick_preprocess(outlier_X, y=y, remove_outliers=True)
        assert X_out.shape[0] < outlier_X.shape[0]


class TestAnalyzeDataQuality:

    def test_returns_dict(self, sample_X):
        report = analyze_data_quality(sample_X)
        assert isinstance(report, dict)
        assert "n_samples" in report
        assert "n_features" in report

    def test_nan_detection(self, nan_X):
        report = analyze_data_quality(nan_X)
        assert report["total_nan"] == 4
        assert report["cols_with_nan"] > 0

    def test_outlier_counts(self, outlier_X):
        report = analyze_data_quality(outlier_X)
        assert report["outliers_iqr"] > 0

    def test_with_feature_names(self, nan_X):
        names = ["f1", "f2", "f3", "f4"]
        report = analyze_data_quality(nan_X, feature_names=names)
        assert "top_nan_features" in report
