# utils/preprocessor.py
# ═══════════════════════════════════════════════════════════
#  Bio-ML Agent — Veri Ön İşleme Pipeline
#
#  Kullanım:
#    from utils.preprocessor import DataPreprocessor
#
#    pp = DataPreprocessor()
#    X_clean = pp.fit_transform(X_train, y_train)
#    X_test_clean = pp.transform(X_test)
#    summary = pp.summary()
# ═══════════════════════════════════════════════════════════

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    PolynomialFeatures,
    StandardScaler,
)

warnings.filterwarnings("ignore")
log = logging.getLogger("bio_ml_agent")


# ─────────────────────────────────────────────
#  Outlier Tespit Yöntemleri
# ─────────────────────────────────────────────

def detect_outliers_iqr(
    X: np.ndarray, factor: float = 1.5
) -> np.ndarray:
    """IQR yöntemiyle outlier tespiti.

    Args:
        X: Özellik matrisi (n_samples, n_features).
        factor: IQR çarpanı (varsayılan 1.5).

    Returns:
        Boolean maske (True = outlier olan satır).
    """
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    outlier_mask = np.any((X < lower) | (X > upper), axis=1)
    return outlier_mask


def detect_outliers_zscore(
    X: np.ndarray, threshold: float = 3.0
) -> np.ndarray:
    """Z-score yöntemiyle outlier tespiti.

    Args:
        X: Özellik matrisi.
        threshold: Z-score eşiği (varsayılan 3.0).

    Returns:
        Boolean maske (True = outlier olan satır).
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1  # sıfır bölme koruması
    z_scores = np.abs((X - mean) / std)
    outlier_mask = np.any(z_scores > threshold, axis=1)
    return outlier_mask


# ─────────────────────────────────────────────
#  Ana Ön İşleme Sınıfı
# ─────────────────────────────────────────────

class DataPreprocessor:
    """Kapsamlı veri ön işleme pipeline'ı.

    Desteklenen işlemler:
      - Eksik değer doldurma (mean, median, most_frequent, constant)
      - Outlier tespiti ve temizleme (IQR, Z-score)
      - Ölçeklendirme (StandardScaler, MinMaxScaler)
      - Polinom özellik üretme
      - PCA boyut indirgeme

    Kullanım:
        pp = DataPreprocessor(
            impute_strategy="median",
            scale_method="standard",
            detect_outliers="iqr",
            pca_components=10,
        )
        X_train_clean = pp.fit_transform(X_train, y_train)
        X_test_clean = pp.transform(X_test)
    """

    def __init__(
        self,
        # Eksik değer
        impute_strategy: str = "median",
        impute_fill_value: Any = 0,
        # Outlier
        detect_outliers: Optional[str] = None,
        outlier_threshold: float = 1.5,
        remove_outliers: bool = False,
        # Ölçeklendirme
        scale_method: Optional[str] = "standard",
        # Feature Engineering
        poly_degree: int = 0,
        poly_interaction_only: bool = False,
        # PCA
        pca_components: Optional[int] = None,
        pca_variance_ratio: Optional[float] = None,
    ):
        # Imputation
        self.impute_strategy = impute_strategy
        self.impute_fill_value = impute_fill_value

        # Outliers
        self.detect_outliers = detect_outliers  # "iqr" | "zscore" | None
        self.outlier_threshold = outlier_threshold
        self.remove_outliers = remove_outliers

        # Scaling
        self.scale_method = scale_method  # "standard" | "minmax" | None

        # Polynomial
        self.poly_degree = poly_degree
        self.poly_interaction_only = poly_interaction_only

        # PCA
        self.pca_components = pca_components
        self.pca_variance_ratio = pca_variance_ratio

        # Internal state
        self._imputer: Optional[SimpleImputer] = None
        self._scaler: Optional[Any] = None
        self._poly: Optional[PolynomialFeatures] = None
        self._pca: Optional[PCA] = None
        self._is_fitted = False

        # Statistics
        self._stats: Dict[str, Any] = {}

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "DataPreprocessor":
        """Pipeline'ı eğitim verisine uydur.

        Args:
            X: Özellik matrisi (n_samples, n_features).
            y: Hedef vektör (opsiyonel, outlier çıkarma için).

        Returns:
            self (fluent API).
        """
        X = np.asarray(X, dtype=float)
        self._stats["original_shape"] = X.shape
        self._stats["nan_count_before"] = int(np.isnan(X).sum())

        # 1. Imputation fit
        if np.isnan(X).any():
            self._imputer = SimpleImputer(
                strategy=self.impute_strategy,
                fill_value=self.impute_fill_value if self.impute_strategy == "constant" else None,
            )
            X = self._imputer.fit_transform(X)
            self._stats["imputed"] = True
        else:
            self._stats["imputed"] = False

        # 2. Outlier detection (sadece istatistik, çıkarma fit_transform'da yapılır)
        if self.detect_outliers:
            if self.detect_outliers == "iqr":
                mask = detect_outliers_iqr(X, self.outlier_threshold)
            elif self.detect_outliers == "zscore":
                mask = detect_outliers_zscore(X, self.outlier_threshold)
            else:
                mask = np.zeros(X.shape[0], dtype=bool)
            self._stats["outlier_count"] = int(mask.sum())
            self._stats["outlier_ratio"] = round(mask.sum() / len(mask), 4)

        # 3. Scaler fit
        if self.scale_method == "standard":
            self._scaler = StandardScaler()
            self._scaler.fit(X)
        elif self.scale_method == "minmax":
            self._scaler = MinMaxScaler()
            self._scaler.fit(X)

        # 4. Polynomial fit
        if self.poly_degree >= 2:
            self._poly = PolynomialFeatures(
                degree=self.poly_degree,
                interaction_only=self.poly_interaction_only,
                include_bias=False,
            )
            X_scaled = self._scaler.transform(X) if self._scaler else X
            self._poly.fit(X_scaled)
            X_poly = self._poly.transform(X_scaled)
            self._stats["poly_features"] = X_poly.shape[1]
        else:
            X_poly = self._scaler.transform(X) if self._scaler else X

        # 5. PCA fit
        if self.pca_components or self.pca_variance_ratio:
            n_components = self.pca_components
            if self.pca_variance_ratio and not self.pca_components:
                n_components = self.pca_variance_ratio
            self._pca = PCA(n_components=n_components)
            self._pca.fit(X_poly)
            self._stats["pca_components"] = self._pca.n_components_
            self._stats["pca_explained_variance"] = round(
                float(self._pca.explained_variance_ratio_.sum()), 4
            )

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Fit edilmiş pipeline ile veriyi dönüştür.

        Args:
            X: Özellik matrisi.

        Returns:
            Dönüştürülmüş özellik matrisi.
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline henüz fit edilmedi. Önce fit() veya fit_transform() çağırın.")

        X = np.asarray(X, dtype=float)

        # 1. Imputation
        if self._imputer is not None:
            X = self._imputer.transform(X)

        # 2. Scaling
        if self._scaler is not None:
            X = self._scaler.transform(X)

        # 3. Polynomial
        if self._poly is not None:
            X = self._poly.transform(X)

        # 4. PCA
        if self._pca is not None:
            X = self._pca.transform(X)

        return X

    def fit_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Fit ve transform'u bir arada yapar.

        Eğer remove_outliers=True ise outlier'lar çıkarılır ve (X_clean, y_clean) döner.
        Aksi halde sadece X_clean döner.

        Args:
            X: Özellik matrisi.
            y: Hedef vektör (opsiyonel).

        Returns:
            X_clean veya (X_clean, y_clean) tuple'ı.
        """
        X = np.asarray(X, dtype=float)

        # Outlier çıkarma (fit_transform'dan önce)
        if self.detect_outliers and self.remove_outliers:
            if self.detect_outliers == "iqr":
                # Önce NaN doldur
                temp_imputer = SimpleImputer(strategy=self.impute_strategy)
                X_temp = temp_imputer.fit_transform(X) if np.isnan(X).any() else X
                mask = detect_outliers_iqr(X_temp, self.outlier_threshold)
            elif self.detect_outliers == "zscore":
                temp_imputer = SimpleImputer(strategy=self.impute_strategy)
                X_temp = temp_imputer.fit_transform(X) if np.isnan(X).any() else X
                mask = detect_outliers_zscore(X_temp, self.outlier_threshold)
            else:
                mask = np.zeros(X.shape[0], dtype=bool)

            clean_mask = ~mask
            X = X[clean_mask]
            self._stats["outlier_removed"] = int(mask.sum())
            if y is not None:
                y = np.asarray(y)[clean_mask]

        self.fit(X, y)
        X_transformed = self.transform(X)

        self._stats["final_shape"] = X_transformed.shape

        if y is not None and self.remove_outliers:
            return X_transformed, y
        return X_transformed

    def summary(self) -> Dict[str, Any]:
        """Pipeline istatistiklerinin özetini döndür.

        Returns:
            İstatistik sözlüğü.
        """
        return {
            "is_fitted": self._is_fitted,
            "impute_strategy": self.impute_strategy,
            "scale_method": self.scale_method,
            "detect_outliers": self.detect_outliers,
            "poly_degree": self.poly_degree if self.poly_degree >= 2 else None,
            "pca_components": self.pca_components,
            **self._stats,
        }

    def summary_text(self) -> str:
        """İnsan okunabilir özet metni.

        Returns:
            Türkçe özet string.
        """
        s = self.summary()
        lines = ["📊 Veri Ön İşleme Özeti", "─" * 40]

        if "original_shape" in s:
            lines.append(f"  Orijinal boyut : {s['original_shape'][0]} satır × {s['original_shape'][1]} özellik")
        if s.get("nan_count_before", 0) > 0:
            lines.append(f"  Eksik değer    : {s['nan_count_before']} (strateji: {s['impute_strategy']})")
        if "outlier_count" in s:
            lines.append(f"  Outlier tespit : {s['outlier_count']} ({s.get('outlier_ratio', 0):.1%})")
        if "outlier_removed" in s:
            lines.append(f"  Outlier çıkarıldı: {s['outlier_removed']} satır")
        if s.get("scale_method"):
            lines.append(f"  Ölçeklendirme  : {s['scale_method']}")
        if s.get("poly_degree"):
            lines.append(f"  Polinom derece : {s['poly_degree']} → {s.get('poly_features', '?')} özellik")
        if "pca_components" in s:
            lines.append(f"  PCA bileşen    : {s['pca_components']} (açıklanan varyans: {s.get('pca_explained_variance', '?'):.1%})")
        if "final_shape" in s:
            lines.append(f"  Son boyut      : {s['final_shape'][0]} satır × {s['final_shape'][1]} özellik")

        return "\n".join(lines)


# ─────────────────────────────────────────────
#  Yardımcı Fonksiyonlar
# ─────────────────────────────────────────────

def quick_preprocess(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    scale: bool = True,
    remove_outliers: bool = False,
    pca: Optional[int] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Hızlı ön işleme — tek satırda kullanım.

    Args:
        X: Özellik matrisi.
        y: Hedef vektör (opsiyonel).
        scale: StandardScaler uygulansın mı.
        remove_outliers: IQR ile outlier çıkarsın mı.
        pca: PCA bileşen sayısı (None = PCA yok).

    Returns:
        X_clean veya (X_clean, y_clean).

    Kullanım:
        X_clean = quick_preprocess(X_train, scale=True, pca=10)
    """
    pp = DataPreprocessor(
        impute_strategy="median",
        scale_method="standard" if scale else None,
        detect_outliers="iqr" if remove_outliers else None,
        remove_outliers=remove_outliers,
        pca_components=pca,
    )
    return pp.fit_transform(X, y)


def analyze_data_quality(X: np.ndarray, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Veri kalitesi analizi.

    Args:
        X: Özellik matrisi.
        feature_names: Özellik adları.

    Returns:
        Kalite raporu sözlüğü.
    """
    X = np.asarray(X, dtype=float)
    n_samples, n_features = X.shape

    # NaN analizi
    nan_per_col = np.isnan(X).sum(axis=0)
    nan_per_row = np.isnan(X).sum(axis=1)

    # Outlier analizi (IQR)
    X_clean = SimpleImputer(strategy="median").fit_transform(X) if np.isnan(X).any() else X
    outlier_mask_iqr = detect_outliers_iqr(X_clean)
    outlier_mask_z = detect_outliers_zscore(X_clean)

    report = {
        "n_samples": n_samples,
        "n_features": n_features,
        "total_nan": int(np.isnan(X).sum()),
        "nan_ratio": round(float(np.isnan(X).sum()) / (n_samples * n_features), 4),
        "cols_with_nan": int((nan_per_col > 0).sum()),
        "rows_with_nan": int((nan_per_row > 0).sum()),
        "outliers_iqr": int(outlier_mask_iqr.sum()),
        "outliers_zscore": int(outlier_mask_z.sum()),
        "constant_features": int((np.nanstd(X, axis=0) == 0).sum()),
    }

    # En çok NaN olan sütunlar
    if feature_names and nan_per_col.sum() > 0:
        top_nan = sorted(
            zip(feature_names, nan_per_col.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )[:5]
        report["top_nan_features"] = [
            {"feature": f, "nan_count": c} for f, c in top_nan if c > 0
        ]

    return report
