import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import json
import warnings

warnings.filterwarnings('ignore')

def _convert_numpy_types(obj):
    """
    Recursively converts NumPy numeric types within an object (dict, list)
    to standard Python numeric types (int, float) for JSON serialization.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(elem) for elem in obj]
    else:
        return obj

def analyze_data_quality(df, target_column=None, feature_names=None):
    """
    Generates a data quality report including missing values, duplicates,
    outliers (for numerical), and basic statistics.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str, optional): Name of the target column. Defaults to None.
        feature_names (list, optional): List of feature names to analyze. If None, all columns except target_column are used.

    Returns:
        dict: A dictionary containing the data quality report.
    """
    report = {}

    if feature_names is None:
        if target_column and target_column in df.columns:
            features_df = df.drop(columns=[target_column])
        else:
            features_df = df
    else:
        features_df = df[feature_names]

    report['Total Samples'] = df.shape[0]
    report['Total Features'] = features_df.shape[1]
    report['Duplicate Rows'] = df.duplicated().sum()

    # Missing Values
    missing_data = features_df.isnull().sum()
    missing_percentage = (features_df.isnull().sum() / df.shape[0]) * 100
    missing_info = pd.DataFrame({'Missing Count': missing_data, 'Missing Percentage': missing_percentage})
    missing_info = missing_info[missing_info['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)
    report['Missing Values'] = missing_info.to_dict(orient='index')

    # Numerical Features Analysis (basic stats and outliers)
    numerical_features = features_df.select_dtypes(include=np.number)
    report['Numerical Features Summary'] = numerical_features.describe().to_dict()

    outlier_info = {}
    for col in numerical_features.columns:
        Q1 = numerical_features[col].quantile(0.25)
        Q3 = numerical_features[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = numerical_features[(numerical_features[col] < lower_bound) | (numerical_features[col] > upper_bound)][col]
        if not outliers.empty:
            outlier_info[col] = {
                'count': int(outliers.count()),
                'percentage': float(outliers.count() / df.shape[0] * 100),
                'min_val': float(outliers.min()),
                'max_val': float(outliers.max())
            }
    report['Outliers (IQR Method)'] = outlier_info

    # Categorical Features Analysis (top categories)
    categorical_features = features_df.select_dtypes(include='object') # Assumes object type for categorical
    categorical_summary = {}
    for col in categorical_features.columns:
        top_categories = features_df[col].value_counts(normalize=True).head(5)
        categorical_summary[col] = top_categories.to_dict()
    report['Categorical Features Summary'] = categorical_summary

    # Ensure all numpy types are converted for JSON serialization
    final_report = _convert_numpy_types(report)
    
    return final_report

def quick_preprocess(df, target_column=None, scale=True, pca=None, impute_strategy='median'):
    """
    Performs quick preprocessing steps on a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str, optional): The name of the target column.
        scale (bool): Whether to scale numerical features. Defaults to True.
        pca (int, optional): Number of PCA components. If None, PCA is not applied.
        impute_strategy (str): Strategy for imputation ('mean', 'median', 'most_frequent').

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    df_processed = df.copy()

    # Separate features and target if target_column is provided
    if target_column and target_column in df_processed.columns:
        X = df_processed.drop(columns=[target_column])
        y = df_processed[target_column]
    else:
        X = df_processed
        y = None

    # Imputation
    numerical_cols = X.select_dtypes(include=np.number).columns
    imputer = SimpleImputer(strategy=impute_strategy)
    X[numerical_cols] = imputer.fit_transform(X[numerical_cols])

    # Scaling
    if scale:
        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    # PCA
    if pca is not None:
        pca_model = PCA(n_components=pca)
        X_pca = pca_model.fit_transform(X[numerical_cols])
        X = pd.DataFrame(X_pca, columns=[f'PCA_{i+1}' for i in range(pca)], index=X.index)

    if y is not None:
        X[target_column] = y
        return X
    return X


class DataPreprocessor:
    """
    A comprehensive data preprocessor for various data cleaning and transformation tasks.
    """
    def __init__(self, impute_strategy="median", scale_method="standard",
                 detect_outliers=None, remove_outliers=False, pca_components=None):
        self.impute_strategy = impute_strategy
        self.scale_method = scale_method
        self.detect_outliers = detect_outliers # 'iqr', 'isolation_forest'
        self.remove_outliers = remove_outliers
        self.pca_components = pca_components

        self.imputer = None
        self.scaler = None
        self.pca_model = None
        self.outlier_detector = None
        self.summary = {}

    def fit(self, X, y=None):
        self._fit_imputer(X)
        X_imputed = self._transform_imputer(X)

        self._fit_scaler(X_imputed)
        X_scaled = self._transform_scaler(X_imputed)

        if self.pca_components is not None:
            self._fit_pca(X_scaled)
        
        # Outlier detection and removal should be handled on X_imputed before scaling/PCA
        # to ensure original value context for detection, but removal can affect indices.
        # For simplicity, we detect and log, removal in transform.
        if self.detect_outliers:
            if self.detect_outliers == 'iqr':
                self.summary['outlier_method'] = 'IQR'
            elif self.detect_outliers == 'isolation_forest':
                self.outlier_detector = IsolationForest(random_state=42)
                self.outlier_detector.fit(X_imputed.select_dtypes(include=np.number))
                self.summary['outlier_method'] = 'Isolation Forest'

        self.summary['impute_strategy'] = self.impute_strategy
        self.summary['scale_method'] = self.scale_method
        self.summary['pca_components'] = self.pca_components
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        y_transformed = y.copy() if y is not None else None

        # Imputation
        if self.imputer:
            numerical_cols = X_transformed.select_dtypes(include=np.number).columns
            X_transformed[numerical_cols] = self._transform_imputer(X_transformed)

        # Outlier Removal (if enabled)
        if self.remove_outliers and self.detect_outliers:
            if self.detect_outliers == 'iqr':
                X_transformed, y_transformed = self._remove_outliers_iqr(X_transformed, y_transformed)
            elif self.detect_outliers == 'isolation_forest' and self.outlier_detector:
                X_transformed, y_transformed = self._remove_outliers_isolation_forest(X_transformed, y_transformed)

        # Scaling
        if self.scaler:
            numerical_cols = X_transformed.select_dtypes(include=np.number).columns
            X_transformed[numerical_cols] = self._transform_scaler(X_transformed[numerical_cols])

        # PCA
        if self.pca_model:
            X_transformed = pd.DataFrame(self.pca_model.transform(X_transformed),
                                         columns=[f'PCA_{i+1}' for i in range(self.pca_components)],
                                         index=X_transformed.index)
        return X_transformed, y_transformed
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

    def _fit_imputer(self, X):
        numerical_cols = X.select_dtypes(include=np.number).columns
        if not numerical_cols.empty:
            self.imputer = SimpleImputer(strategy=self.impute_strategy)
            self.imputer.fit(X[numerical_cols])
            self.summary['imputed_features'] = numerical_cols.tolist()
            self.summary['imputation_stats'] = {col: self.imputer.statistics_[i] for i, col in enumerate(numerical_cols)}
        else:
            self.imputer = None

    def _transform_imputer(self, X):
        X_copy = X.copy()
        if self.imputer:
            numerical_cols = X_copy.select_dtypes(include=np.number).columns
            X_copy[numerical_cols] = self.imputer.transform(X_copy[numerical_cols])
        return X_copy

    def _fit_scaler(self, X):
        numerical_cols = X.select_dtypes(include=np.number).columns
        if not numerical_cols.empty:
            if self.scale_method == "standard":
                self.scaler = StandardScaler()
            elif self.scale_method == "minmax":
                self.scaler = MinMaxScaler()
            elif self.scale_method == "robust":
                self.scaler = RobustScaler()
            else:
                self.scaler = None
            
            if self.scaler:
                self.scaler.fit(X[numerical_cols])
                self.summary['scaled_features'] = numerical_cols.tolist()
        else:
            self.scaler = None

    def _transform_scaler(self, X_numerical):
        if self.scaler:
            return self.scaler.transform(X_numerical)
        return X_numerical

    def _fit_pca(self, X):
        self.pca_model = PCA(n_components=self.pca_components)
        self.pca_model.fit(X)
        self.summary['pca_explained_variance_ratio'] = self.pca_model.explained_variance_ratio_.tolist()

    def _remove_outliers_iqr(self, X, y):
        # This function should remove rows containing outliers
        initial_rows = X.shape[0]
        rows_to_keep = pd.Series(True, index=X.index)
        
        numerical_cols = X.select_dtypes(include=np.number).columns
        for col in numerical_cols:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identify outliers in this column
            col_outliers = (X[col] < lower_bound) | (X[col] > upper_bound)
            rows_to_keep = rows_to_keep & (~col_outliers) # Mark rows with outliers in this column for removal
        
        X_clean = X[rows_to_keep]
        y_clean = y[rows_to_keep] if y is not None else None
        removed_rows = initial_rows - X_clean.shape[0]
        self.summary['iqr_outliers_removed_count'] = removed_rows
        self.summary['iqr_outliers_removed_percentage'] = (removed_rows / initial_rows) * 100 if initial_rows > 0 else 0
        return X_clean, y_clean

    def _remove_outliers_isolation_forest(self, X, y):
        # Isolation Forest outputs -1 for outliers and 1 for inliers
        if self.outlier_detector:
            initial_rows = X.shape[0]
            numerical_X = X.select_dtypes(include=np.number)
            outlier_predictions = self.outlier_detector.predict(numerical_X)
            
            X_clean = X[outlier_predictions == 1]
            y_clean = y[outlier_predictions == 1] if y is not None else None
            
            removed_rows = initial_rows - X_clean.shape[0]
            self.summary['iso_forest_outliers_removed_count'] = removed_rows
            self.summary['iso_forest_outliers_removed_percentage'] = (removed_rows / initial_rows) * 100 if initial_rows > 0 else 0
            return X_clean, y_clean
        return X, y # Return original if no detector

    def summary_text(self):
        summary_str = "--- Data Preprocessor Summary ---\n"
        for key, value in self.summary.items():
            if isinstance(value, dict):
                summary_str += f"{key.replace('_', ' ').title()}:\n"
                for sub_key, sub_value in value.items():
                    summary_str += f"  {sub_key.replace('_', ' ').title()}: {sub_value}\n"
            elif isinstance(value, list) and key != 'pca_explained_variance_ratio':
                 summary_str += f"{key.replace('_', ' ').title()}: {', '.join(map(str, value))}\n"
            else:
                summary_str += f"{key.replace('_', ' ').title()}: {value}\n"
        
        if 'pca_explained_variance_ratio' in self.summary:
            summary_str += "PCA Explained Variance Ratio:\n"
            for i, ratio in enumerate(self.summary['pca_explained_variance_ratio']):
                summary_str += f"  Component {i+1}: {ratio:.4f}\n"

        return summary_str