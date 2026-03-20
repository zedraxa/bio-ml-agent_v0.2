import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from collections import Counter
import warnings

warnings.filterwarnings('ignore', category=UserWarning) # Suppress specific warnings

class DataPreprocessor:
    def __init__(self, impute_strategy='median', scale_method='standard',
                 detect_outliers=None, remove_outliers=False, pca_components=None,
                 random_state=42):
        self.impute_strategy = impute_strategy
        self.scale_method = scale_method
        self.detect_outliers = detect_outliers # 'iqr', 'zscore'
        self.remove_outliers = remove_outliers
        self.pca_components = pca_components
        self.random_state = random_state

        self.imputer = None
        self.scaler = None
        self.pca = None
        self.summary_log = []
        self.original_columns = None
        self.removed_outliers_count = 0

    def _log(self, message):
        self.summary_log.append(message)
        print(message)

    def fit(self, X, y=None):
        self.original_columns = X.columns if isinstance(X, pd.DataFrame) else [f'col_{i}' for i in range(X.shape[1])]
        
        X_processed = X.copy()
        
        # 1. Imputation
        if X_processed.isnull().sum().sum() > 0:
            self._log(f"Eksik değerler tespit edildi. {self.impute_strategy} stratejisi ile dolduruluyor.")
            self.imputer = SimpleImputer(strategy=self.impute_strategy)
            X_processed[X_processed.columns] = self.imputer.fit_transform(X_processed)
        else:
            self._log("Eksik değer bulunamadı.")

        # 2. Outlier Detection and Removal (fit only for detection parameters)
        if self.detect_outliers and self.remove_outliers:
            self._log(f"Aykırı değerler tespit ediliyor ve kaldırılıyor ({self.detect_outliers} metodu).")
            outlier_indices = self._detect_outliers(X_processed)
            self.removed_outliers_count = len(outlier_indices)
            if self.removed_outliers_count > 0:
                self._log(f"{self.removed_outliers_count} aykırı değer gözlemi kaldırılacak.")
            else:
                self._log("Aykırı değer bulunamadı veya kaldırılmadı.")
        
        # 3. Scaling
        if self.scale_method:
            self._log(f"Özellikler {self.scale_method} metodu ile ölçeklendiriliyor.")
            if self.scale_method == 'standard':
                self.scaler = StandardScaler()
            elif self.scale_method == 'minmax':
                self.scaler = MinMaxScaler()
            elif self.scale_method == 'robust':
                self.scaler = RobustScaler()
            else:
                raise ValueError("Geçersiz ölçeklendirme metodu. 'standard', 'minmax' veya 'robust' olmalı.")
            X_processed[X_processed.columns] = self.scaler.fit_transform(X_processed)
        else:
            self._log("Ölçeklendirme yapılmayacak.")

        # 4. PCA
        if self.pca_components:
            self._log(f"PCA uygulanıyor, {self.pca_components} bileşene düşürülecek.")
            self.pca = PCA(n_components=self.pca_components, random_state=self.random_state)
            self.pca.fit(X_processed) # Fit PCA on scaled data
            self._log(f"Açıklanan varyans oranı: {self.pca.explained_variance_ratio_.sum():.2f}")
        else:
            self._log("PCA uygulanmayacak.")

        return self

    def transform(self, X, y=None):
        if self.original_columns is not None and isinstance(X, pd.DataFrame):
            # Ensure columns are in the same order as during fit
            X = X[self.original_columns].copy()
        else:
            X = X.copy()

        # 1. Imputation
        if self.imputer:
            X[X.columns] = self.imputer.transform(X)

        # 2. Outlier Removal (only if remove_outliers is True and fit detected outliers)
        # Outlier removal should primarily happen on training data to avoid data leakage.
        # For test data, we just transform, not remove based on test data outliers.
        # If the user wants to remove outliers from the test set as well (based on train distribution),
        # this logic needs to be more complex. For now, assume removal is only for train.
        # If remove_outliers is True, we assume fit already handled removal for train.
        # For transform, we apply the learned transformations, not new removals based on test outliers.

        # 3. Scaling
        if self.scaler:
            X[X.columns] = self.scaler.transform(X)

        # 4. PCA
        if self.pca:
            X = pd.DataFrame(self.pca.transform(X), index=X.index)
            X.columns = [f'PCA_{i+1}' for i in range(X.shape[1])]

        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        X_transformed = self.transform(X)

        if self.remove_outliers and self.removed_outliers_count > 0:
            # We need to re-fit and re-transform *after* outlier removal if it affects the data used for scaler/pca.
            # A more robust way: first remove outliers from X, then fit imputer/scaler/pca on the cleaned X.
            # Current implementation detects outliers from the *initial* X_processed, but the transformations are applied sequentially.
            # Let's refine:
            X_copy_for_removal = X.copy()
            y_copy_for_removal = y.copy() if y is not None else None

            if X_copy_for_removal.isnull().sum().sum() > 0:
                self.imputer = SimpleImputer(strategy=self.impute_strategy)
                X_copy_for_removal[X_copy_for_removal.columns] = self.imputer.fit_transform(X_copy_for_removal)

            outlier_indices = self._detect_outliers(X_copy_for_removal)
            if len(outlier_indices) > 0:
                self._log(f"{len(outlier_indices)} aykırı değer gözlemi eğitim setinden kaldırılıyor.")
                X_transformed = X_copy_for_removal.drop(outlier_indices).reset_index(drop=True)
                if y_copy_for_removal is not None:
                    y_transformed = y_copy_for_removal.drop(outlier_indices).reset_index(drop=True)
                else:
                    y_transformed = None
                self._log(f"Kaldırma sonrası eğitim seti boyutu: {X_transformed.shape[0]}")
            else:
                X_transformed = X_copy_for_removal
                y_transformed = y_copy_for_removal

            # Now fit scaler and PCA on the (possibly) outlier-removed data
            if self.scale_method:
                if self.scale_method == 'standard': self.scaler = StandardScaler()
                elif self.scale_method == 'minmax': self.scaler = MinMaxScaler()
                elif self.scale_method == 'robust': self.scaler = RobustScaler()
                X_transformed[X_transformed.columns] = self.scaler.fit_transform(X_transformed)

            if self.pca_components:
                self.pca = PCA(n_components=self.pca_components, random_state=self.random_state)
                X_transformed = pd.DataFrame(self.pca.fit_transform(X_transformed), index=X_transformed.index)
                X_transformed.columns = [f'PCA_{i+1}' for i in range(X_transformed.shape[1])]
            
            return X_transformed, y_transformed if y is not None else X_transformed

        return X_transformed, y

    def _detect_outliers(self, X_df):
        outlier_indices = set()
        for col in X_df.columns:
            if pd.api.types.is_numeric_dtype(X_df[col]):
                if self.detect_outliers == 'iqr':
                    Q1 = X_df[col].quantile(0.25)
                    Q3 = X_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    col_outliers = X_df[(X_df[col] < lower_bound) | (X_df[col] > upper_bound)].index
                    outlier_indices.update(col_outliers)
                elif self.detect_outliers == 'zscore':
                    mean = X_df[col].mean()
                    std = X_df[col].std()
                    if std == 0: continue # Avoid division by zero
                    z_scores = (X_df[col] - mean) / std
                    col_outliers = X_df[np.abs(z_scores) > 3].index # Z-score threshold of 3
                    outlier_indices.update(col_outliers)
        return list(outlier_indices)

    def summary_text(self):
        return "\n".join(self.summary_log)

def analyze_data_quality(df, feature_names=None):
    report_lines = ["--- Veri Kalite Raporu ---"]
    
    report_lines.append(f"Toplam Satır: {df.shape[0]}, Toplam Sütun: {df.shape[1]}")
    
    # Missing Values
    missing_info = df.isnull().sum()
    missing_info = missing_info[missing_info > 0]
    if not missing_info.empty:
        report_lines.append("\nEksik Değerler:")
        for col, count in missing_info.items():
            report_lines.append(f"- {col}: {count} ({count/df.shape[0]:.2%})")
    else:
        report_lines.append("\nEksik değer bulunamadı.")

    # Duplicate Rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        report_lines.append(f"\nTekrar Eden Satırlar: {duplicates} ({duplicates/df.shape[0]:.2%})")
    else:
        report_lines.append("\nTekrar eden satır bulunamadı.")

    # Data Types
    report_lines.append("\nVeri Tipleri:")
    for col, dtype in df.dtypes.items():
        report_lines.append(f"- {col}: {dtype}")

    # Unique Values for Categorical/Low-Cardinality columns
    report_lines.append("\nBenzersiz Değerler (İlk 10):")
    for col in df.columns:
        if df[col].nunique() < 20 and df[col].nunique() > 1: # Consider as categorical or low-cardinality numerical
            report_lines.append(f"- {col} ({df[col].nunique()} benzersiz): {df[col].value_counts().index.tolist()[:10]}")
        elif pd.api.types.is_numeric_dtype(df[col]):
            report_lines.append(f"- {col} (Sayısal): Min={df[col].min():.2f}, Max={df[col].max():.2f}, Mean={df[col].mean():.2f}, Std={df[col].std():.2f}")

    # Basic Statistics for numerical columns
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numerical_cols:
        report_lines.append("\nSayısal Sütun İstatistikleri:")
        report_lines.append(df[numerical_cols].describe().to_string())

    return "\n".join(report_lines)

def quick_preprocess(df, target_column=None, scale=True, pca_components=None, test_size=0.2, random_state=42):
    """
    Hızlı veri ön işleme fonksiyonu.
    Eksik değerleri medyan ile doldurur, isteğe bağlı olarak ölçeklendirir ve PCA uygular,
    ardından eğitim/test setlerine böler.
    """
    X = df.drop(columns=[target_column]) if target_column else df
    y = df[target_column] if target_column else None

    # Imputation
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    # Scaling
    X_scaled = X_imputed
    if scale:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns, index=X_imputed.index)
        print("Veri standart ölçeklendirme ile işlendi.")

    # PCA
    X_pca = X_scaled
    if pca_components:
        pca = PCA(n_components=pca_components, random_state=random_state)
        X_pca = pd.DataFrame(pca.fit_transform(X_scaled), index=X_scaled.index)
        X_pca.columns = [f'PCA_{i+1}' for i in range(X_pca.shape[1])]
        print(f"PCA uygulandı, {pca_components} bileşene düşürüldü. Açıklanan varyans: {pca.explained_variance_ratio_.sum():.2f}")

    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=test_size, random_state=random_state, stratify=y)
        print(f"Veri eğitim (%{(1-test_size)*100:.0f}) ve test (%{test_size*100:.0f}) setlerine bölündü.")
        return X_train, X_test, y_train, y_test, X_pca.columns.tolist()
    else:
        print("Hedef sütun belirtilmedi, sadece işlenmiş X döndürülüyor.")
        return X_pca, X_pca.columns.tolist()