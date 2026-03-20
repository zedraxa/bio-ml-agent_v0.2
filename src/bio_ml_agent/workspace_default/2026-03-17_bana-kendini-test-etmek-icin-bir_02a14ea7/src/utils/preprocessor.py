import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from collections import Counter

class DataPreprocessor:
    def __init__(self, impute_strategy='median', scale_method='standard',
                 detect_outliers=None, remove_outliers=False, pca_components=None):
        self.impute_strategy = impute_strategy
        self.scale_method = scale_method
        self.detect_outliers = detect_outliers
        self.remove_outliers = remove_outliers
        self.pca_components = pca_components

        self.imputer = None
        self.scaler = None
        self.outlier_detector = None
        self.pca = None
        self.summary_log = []

    def fit(self, X, y=None):
        X_processed = X.copy()
        self.summary_log = []

        # 1. Imputation
        self.imputer = SimpleImputer(strategy=self.impute_strategy)
        X_processed = self._impute(X_processed)

        # 2. Outlier Detection/Removal (if specified)
        if self.detect_outliers:
            X_processed, outlier_indices = self._handle_outliers(X_processed)
            if self.remove_outliers and outlier_indices is not None:
                self.summary_log.append(f"Removed {len(outlier_indices)} outliers.")
                # Adjust y if it exists
                if y is not None:
                    y = y.drop(y.index[outlier_indices]).reset_index(drop=True)
                X_processed = X_processed.drop(X_processed.index[outlier_indices]).reset_index(drop=True)


        # 3. Scaling
        self.scaler = StandardScaler() if self.scale_method == 'standard' else None
        if self.scaler:
            X_processed = pd.DataFrame(self.scaler.fit_transform(X_processed), columns=X_processed.columns)
            self.summary_log.append(f"Data scaled using {self.scale_method} method.")

        # 4. PCA
        if self.pca_components:
            self.pca = PCA(n_components=self.pca_components)
            X_processed = pd.DataFrame(self.pca.fit_transform(X_processed),
                                       columns=[f'PC{i+1}' for i in range(self.pca_components)])
            self.summary_log.append(f"Applied PCA with {self.pca_components} components. Explained variance: {self.pca.explained_variance_ratio_.sum():.2f}")

        return self, y # Return y as well, in case outliers were removed

    def transform(self, X):
        X_processed = X.copy()

        if self.imputer:
            X_processed = self._impute(X_processed)

        # Outlier handling is typically not done on test set for removal, only detection
        # If we removed outliers from train, we just transform test without removing.

        if self.scaler:
            X_processed = pd.DataFrame(self.scaler.transform(X_processed), columns=X_processed.columns)

        if self.pca:
            X_processed = pd.DataFrame(self.pca.transform(X_processed),
                                       columns=[f'PC{i+1}' for i in range(self.pca_components)])

        return X_processed

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X), y # y is returned from fit, so use that one

    def _impute(self, X):
        # Identify columns with zeros that should be imputed (e.g., BloodPressure, SkinThickness, Insulin, BMI, Glucose)
        # Assuming 0s in these columns are actually missing values
        cols_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in cols_to_impute:
            if col in X.columns:
                X[col] = X[col].replace(0, np.nan)

        X_imputed = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns, index=X.index)
        self.summary_log.append(f"Imputed missing values (0s replaced with NaN) using '{self.impute_strategy}' strategy.")
        return X_imputed

    def _handle_outliers(self, X):
        outlier_indices = None
        if self.detect_outliers == 'iqr':
            outlier_indices = []
            for col in X.columns:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                col_outliers = X[(X[col] < lower_bound) | (X[col] > upper_bound)].index.tolist()
                outlier_indices.extend(col_outliers)
            outlier_indices = list(set(outlier_indices)) # Remove duplicates
            self.summary_log.append(f"Detected {len(outlier_indices)} outliers using IQR method.")
        elif self.detect_outliers == 'isolation_forest':
            self.outlier_detector = IsolationForest(random_state=42)
            preds = self.outlier_detector.fit_predict(X)
            outlier_indices = X[preds == -1].index.tolist()
            self.summary_log.append(f"Detected {len(outlier_indices)} outliers using Isolation Forest.")

        if self.remove_outliers and outlier_indices:
            return X.drop(outlier_indices).reset_index(drop=True), outlier_indices
        return X, outlier_indices

    def summary_text(self):
        return "\n".join(self.summary_log)

def analyze_data_quality(df, feature_names=None):
    """Generates a comprehensive data quality report."""
    report = ["--- Data Quality Report ---"]
    report.append(f"Total Rows: {len(df)}")
    report.append(f"Total Columns: {len(df.columns)}")
    report.append("\nMissing Values (including 0s in specific columns):")

    missing_info = {}
    for col in df.columns:
        # Check for actual NaNs
        n_nan = df[col].isnull().sum()
        # Check for 0s in specific columns (common for Pima dataset)
        n_zeros_in_critical = 0
        if col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
            n_zeros_in_critical = (df[col] == 0).sum()
        
        total_missing = n_nan + n_zeros_in_critical
        if total_missing > 0:
            missing_info[col] = total_missing
            report.append(f"- {col}: {total_missing} ({total_missing / len(df) * 100:.2f}%)")

    if not missing_info:
        report.append("  No missing values detected (including critical 0s).")

    report.append("\nDescriptive Statistics:")
    report.append(df.describe().to_string())

    report.append("\nData Types:")
    report.append(df.dtypes.to_string())

    report.append("\nUnique Values (Top 5 for categorical/low cardinality):")
    for col in df.columns:
        unique_count = df[col].nunique()
        if unique_count <= 10: # Consider low cardinality if less than 10 unique values
            report.append(f"- {col} ({unique_count} unique): {df[col].value_counts().head(5).index.tolist()}")
        elif unique_count < 0.5 * len(df): # If not too many unique values
            report.append(f"- {col} ({unique_count} unique): {df[col].value_counts().head(3).index.tolist()}...")
        else:
            report.append(f"- {col} ({unique_count} unique): High cardinality (numeric or many categories).")

    report.append("\n--- End Data Quality Report ---")
    return "\n".join(report)

def quick_preprocess(df, target_col=None, scale=True, pca=None, test_size=0.2, random_state=42):
    """
    Performs quick preprocessing steps for a DataFrame.
    Returns X_train, X_test, y_train, y_test if target_col is provided,
    otherwise returns preprocessed DataFrame.
    """
    if target_col:
        X = df.drop(columns=[target_col])
        y = df[target_col]
    else:
        X = df.copy()
        y = None

    pp = DataPreprocessor(impute_strategy="median", scale_method="standard" if scale else None,
                          pca_components=pca, detect_outliers=None, remove_outliers=False)

    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        pp.fit(X_train, y_train)
        X_train_clean = pp.transform(X_train)
        X_test_clean = pp.transform(X_test)
        return X_train_clean, X_test_clean, y_train, y_test
    else:
        pp.fit(X)
        X_clean = pp.transform(X)
        return X_clean