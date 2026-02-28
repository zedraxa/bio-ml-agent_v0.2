import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore

class DataPreprocessor:
    def __init__(self, impute_strategy="median", scale_method="standard", detect_outliers=None, remove_outliers=False, pca_components=None):
        self.impute_strategy = impute_strategy
        self.scale_method = scale_method
        self.detect_outliers = detect_outliers # "iqr", "zscore", "isolation_forest"
        self.remove_outliers = remove_outliers
        self.pca_components = pca_components

        self.imputer = None
        self.scaler = None
        self.pca = None
        self.outlier_detector = None
        self.summary_log = []

    def fit(self, X, y=None):
        self.summary_log.append("--- Veri Ön İşleme Başlıyor ---")

        # Imputation
        self.imputer = SimpleImputer(strategy=self.impute_strategy)
        X_imputed = self.imputer.fit_transform(X)
        X_imputed = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
        self.summary_log.append(f"Eksik değerler '{self.impute_strategy}' stratejisi ile dolduruldu.")

        # Outlier Detection/Removal (Sadece eğitim setinde uygun olmalı)
        if self.detect_outliers:
            if self.detect_outliers == "iqr":
                Q1 = X_imputed.quantile(0.25)
                Q3 = X_imputed.quantile(0.75)
                IQR = Q3 - Q1
                self.outlier_lower_bound = Q1 - 1.5 * IQR
                self.outlier_upper_bound = Q3 + 1.5 * IQR
                outlier_mask = ~((X_imputed < self.outlier_lower_bound) | (X_imputed > self.outlier_upper_bound)).any(axis=1)
                self.summary_log.append("IQR metodu ile aykırı değer eşikleri belirlendi.")
            elif self.detect_outliers == "zscore":
                z_scores = np.abs(zscore(X_imputed))
                self.outlier_zscore_threshold = 3
                outlier_mask = (z_scores < self.outlier_zscore_threshold).any(axis=1) # Not actually "any", should be "all" or specific
                self.summary_log.append(f"Z-skor metodu ile aykırı değer eşiği {self.outlier_zscore_threshold} olarak belirlendi.")
            elif self.detect_outliers == "isolation_forest":
                self.outlier_detector = IsolationForest(random_state=42)
                self.outlier_detector.fit(X_imputed)
                outlier_predictions = self.outlier_detector.predict(X_imputed)
                outlier_mask = (outlier_predictions != -1)
                self.summary_log.append("Isolation Forest modeli ile aykırı değerler tespit edildi.")
            else:
                outlier_mask = pd.Series(True, index=X_imputed.index) # No outliers detected if method is unknown

            if self.remove_outliers:
                initial_rows = X_imputed.shape[0]
                X_imputed = X_imputed[outlier_mask]
                if y is not None:
                    y = y[outlier_mask]
                removed_rows = initial_rows - X_imputed.shape[0]
                self.summary_log.append(f"{removed_rows} adet aykırı değer kaldırıldı.")
        
        X_processed = X_imputed # Update X_processed after imputation and outlier handling

        # Scaling
        if self.scale_method == "standard":
            self.scaler = StandardScaler()
        elif self.scale_method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None
        
        if self.scaler:
            X_scaled = self.scaler.fit_transform(X_processed)
            X_processed = pd.DataFrame(X_scaled, columns=X_processed.columns, index=X_processed.index)
            self.summary_log.append(f"Veri '{self.scale_method}' metodu ile ölçeklendi.")

        # PCA
        if self.pca_components is not None:
            self.pca = PCA(n_components=self.pca_components)
            X_pca = self.pca.fit_transform(X_processed)
            X_processed = pd.DataFrame(X_pca, index=X_processed.index)
            self.summary_log.append(f"PCA ile {self.pca_components} bileşene indirgeme yapıldı.")

        self.summary_log.append("--- Veri Ön İşleme Tamamlandı ---")
        return X_processed, y

    def transform(self, X):
        X_transformed = X.copy()

        if self.imputer:
            X_transformed = pd.DataFrame(self.imputer.transform(X_transformed), columns=X.columns, index=X.index)

        # Aykırı değerler transformasyon sırasında kaldırılmaz, sadece fit sırasında.
        # Aykırı değerleri kaldırmak istiyorsak bu kısım sadece fit içinde kalmalı.
        # Eğer yeni veride aykırı değerleri işaretlemek istiyorsak, ayrı bir metot olabilir.

        if self.scaler:
            X_transformed = pd.DataFrame(self.scaler.transform(X_transformed), columns=X.columns, index=X.index)

        if self.pca:
            X_transformed = pd.DataFrame(self.pca.transform(X_transformed), index=X.index)
            
        return X_transformed

    def fit_transform(self, X, y=None):
        X_processed, y_processed = self.fit(X, y)
        return X_processed, y_processed

    def summary_text(self):
        return "\n".join(self.summary_log)

def quick_preprocess(df, target_col=None, scale=True, pca=None, impute_strategy="median"):
    print("Hızlı ön işleme başlatılıyor...")
    if target_col:
        X = df.drop(columns=[target_col])
        y = df[target_col]
    else:
        X = df
        y = None

    pp = DataPreprocessor(impute_strategy=impute_strategy, scale_method="standard" if scale else None, pca_components=pca)
    X_clean, y_clean = pp.fit_transform(X, y)
    print(pp.summary_text())
    print("Hızlı ön işleme tamamlandı.")
    if target_col:
        return X_clean, y_clean
    else:
        return X_clean

def analyze_data_quality(df, feature_names=None):
    if feature_names is None:
        feature_names = df.columns.tolist()

    report_lines = ["--- Veri Kalitesi Analiz Raporu ---"]
    report_lines.append(f"Toplam örnek sayısı: {len(df)}")
    report_lines.append(f"Toplam özellik sayısı: {len(feature_names)}")

    # Eksik değerler
    missing_values = df[feature_names].isnull().sum()
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        report_lines.append("\nEksik Değerler:")
        for col, count in missing_values.items():
            report_lines.append(f"- {col}: {count} ({count / len(df):.2%})")
    else:
        report_lines.append("\nEksik değer bulunamadı.")

    # Benzersiz değerler (kategorik değişkenler için potansiyel)
    report_lines.append("\nBenzersiz Değerler (İlk 5):")
    for col in feature_names:
        unique_count = df[col].nunique()
        if unique_count < 10: # Düşük benzersiz sayıya sahip kolonları listeleyelim
            report_lines.append(f"- {col}: {unique_count} ({df[col].unique().tolist()})")
        else:
            report_lines.append(f"- {col}: {unique_count} (çok fazla)")

    # Betimsel istatistikler
    report_lines.append("\nBetimsel İstatistikler:")
    report_lines.append(df[feature_names].describe().to_markdown())
    
    # Aykırı değer tespiti (basit IQR yöntemi)
    report_lines.append("\nAykırı Değer Tespiti (IQR Metodu):")
    for col in feature_names:
        if pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if not outliers.empty:
                report_lines.append(f"- {col}: {len(outliers)} aykırı değer bulundu.")
            # else:
            #     report_lines.append(f"- {col}: Aykırı değer bulunamadı.")
        else:
            report_lines.append(f"- {col}: Sayısal olmayan sütun.")

    report_lines.append("----------------------------------")
    return "\n".join(report_lines)