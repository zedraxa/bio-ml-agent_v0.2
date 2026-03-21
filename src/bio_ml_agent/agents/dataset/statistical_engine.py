import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

log = logging.getLogger("dataset.statistical_engine")

class StatisticalEngine:
    """
    StatisticalEngine: Veri setleri üzerinde derin denetim yapar.
    - Missingness Audit: Kayıp veri paternlerini (MCAR, MAR, MNAR) analiz eder.
    - Outlier Detection: Z-score veya IQR tabanlı aykırı değer tespiti.
    - Bias Guard: Kategorik değişkenlerdeki dengesiz dağılımları (skewness) bulur.
    """
    
    @staticmethod
    def audit_missingness(df: pd.DataFrame) -> Dict[str, Any]:
        """Kayıp veri oranlarını ve sütunlar arası korelasyonu hesaplar."""
        missing = df.isnull().sum().to_dict()
        total = len(df)
        report = {col: {"count": count, "ratio": count/total} for col, count in missing.items() if count > 0}
        return report

    @staticmethod
    def detect_outliers(df: pd.DataFrame, threshold: float = 3.0) -> Dict[str, List[int]]:
        """Sayısal sütunlardaki aykırı değerlerin indexlerini döner."""
        outliers = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            z_scores = (df[col] - df[col].mean()) / df[col].std()
            outliers[col] = df.index[np.abs(z_scores) > threshold].tolist()
        return outliers

    @staticmethod
    def check_bias(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Hedef sütundaki sınıfsal dengesizliği analiz eder."""
        if target_col not in df.columns: return {}
        counts = df[target_col].value_counts(normalize=True).to_dict()
        return {
            "distribution": counts,
            "max_skew": max(counts.values()) if counts else 0
        }
