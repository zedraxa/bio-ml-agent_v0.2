import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from scipy import stats

class DataPreprocessor:
    def __init__(self, impute_strategy="median", scale_method="standard", detect_outliers=None, remove_outliers=False, pca_components=None):
        self.impute_strategy = impute_strategy
        self.scale_method = scale_method
        self.detect_outliers = detect_outliers # "iqr", "zscore"
        self.remove_outliers = remove_outliers
        self.pca_components = pca_components
        self.imputer = None
        self.scaler = None
        self.pca = None
        self.outlier_indices = None
        self.summary = {}

    def fit(self, X, y=None):
        self.summary = {}
        X_processed = X.copy()
        
        # 1. Imputation
        if X_processed.isnull().sum().sum() > 0:
            self.imputer = SimpleImputer(strategy=self.impute_strategy)
            X_processed = pd.DataFrame(self.imputer.fit_transform(X_processed), columns=X.columns, index=X.index)
            self.summary['imputation'] = f"Applied {self.impute_strategy} imputation for missing values."
        else:
            self.summary['imputation'] = "No missing values found."

        # 2. Outlier Detection and Removal (fit only, removal done in transform)
        if self.detect_outliers:
            self.outlier_indices = self._detect_outliers(X_processed)
            self.summary['outliers_detected'] = f"Detected {len(self.outlier_indices)} outliers using {self.detect_outliers} method."
        else:
            self.summary['outliers_detected'] = "No outlier detection specified."

        # If removing outliers during fit, also remove from y if provided
        if self.remove_outliers and self.outlier_indices is not None:
            if y is not None:
                y = y.drop(self.outlier_indices, errors='ignore')
            X_processed = X_processed.drop(self.outlier_indices, errors='ignore')
            self.summary['outliers_removed_fit'] = f"Removed {len(self.outlier_indices)} outliers from training data."
            
        # 3. Scaling
        if self.scale_method == "standard":
            self.scaler = StandardScaler()
        elif self.scale_method == "minmax":
            self.scaler = MinMaxScaler()
        elif self.scale_method == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = None

        if self.scaler:
            X_processed = pd.DataFrame(self.scaler.fit_transform(X_processed), columns=X_processed.columns, index=X_processed.index)
            self.summary['scaling'] = f"Applied {self.scale_method} scaling."
        else:
            self.summary['scaling'] = "No scaling applied."

        # 4. PCA
        if self.pca_components:
            self.pca = PCA(n_components=self.pca_components)
            self.pca.fit(X_processed) # Fit PCA after scaling
            self.summary['pca'] = f"Applied PCA with {self.pca_components} components. Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.2f}"
        else:
            self.summary['pca'] = "No PCA applied."

        return self

    def transform(self, X, y=None):
        X_processed = X.copy()
        y_processed = y.copy() if y is not None else None

        # 1. Imputation
        if self.imputer:
            X_processed = pd.DataFrame(self.imputer.transform(X_processed), columns=X.columns, index=X.index)

        # 2. Outlier Removal (using indices from fit)
        if self.remove_outliers and self.outlier_indices is not None:
            original_indices = X_processed.index
            # Find which of the test set rows match the outlier indices detected during fit
            test_outlier_indices = original_indices.intersection(self.outlier_indices)
            if not test_outlier_indices.empty:
                X_processed = X_processed.drop(test_outlier_indices, errors='ignore')
                if y_processed is not None:
                    y_processed = y_processed.drop(test_outlier_indices, errors='ignore')
                self.summary['outliers_removed_transform'] = f"Removed {len(test_outlier_indices)} outliers from transformed data."

        # 3. Scaling
        if self.scaler:
            X_processed = pd.DataFrame(self.scaler.transform(X_processed), columns=X_processed.columns, index=X_processed.index)

        # 4. PCA
        if self.pca:
            X_processed = pd.DataFrame(self.pca.transform(X_processed), index=X_processed.index)
            # Generate PCA feature names
            pca_feature_names = [f'PCA_Component_{i+1}' for i in range(X_processed.shape[1])]
            X_processed.columns = pca_feature_names
            
        if y is not None:
            return X_processed, y_processed
        return X_processed

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

    def _detect_outliers(self, df):
        outlier_indices = pd.Index([])
        if self.detect_outliers == "iqr":
            for col in df.select_dtypes(include=np.number).columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
                outlier_indices = outlier_indices.union(col_outliers)
        elif self.detect_outliers == "zscore":
            for col in df.select_dtypes(include=np.number).columns:
                z_scores = np.abs(stats.zscore(df[col]))
                col_outliers = df[z_scores > 3].index # Z-score threshold of 3
                outlier_indices = outlier_indices.union(col_outliers)
        return outlier_indices

    def summary_text(self):
        s = "--- Data Preprocessor Summary ---\n"
        for k, v in self.summary.items():
            s += f"- {k.replace('_', ' ').capitalize()}: {v}\n"
        s += "--------------------------------\n"
        return s

def quick_preprocess(df, target_column=None, scale=True, pca=None, test_size=0.2, random_state=42):
    """
    Hızlı veri önişleme, hedef sütunu ayırma, ölçekleme ve isteğe bağlı PCA.
    """
    if target_column:
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        X = df
        y = None

    if scale:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    if pca:
        pca_model = PCA(n_components=pca)
        X = pd.DataFrame(pca_model.fit_transform(X), index=X.index)
        pca_feature_names = [f'PCA_Component_{i+1}' for i in range(X.shape[1])]
        X.columns = pca_feature_names
        print(f"PCA ile {pca} bileşene düşürüldü. Açıklanan varyans oranı: {pca_model.explained_variance_ratio_.sum():.2f}")

    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() > 1 else None)
        return X_train, X_test, y_train, y_test
    return X

def analyze_data_quality(df, feature_names=None):
    """
    Veri kalitesi raporu oluşturur.
    """
    report = "--- Data Quality Report ---\n"
    report += f"Total rows: {len(df)}\n"
    report += f"Total columns: {len(df.columns)}\n\n"

    # Missing values
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    if not missing_data.empty:
        report += "Missing Values:\n"
        for col, count in missing_data.items():
            report += f"  - {col}: {count} ({count / len(df):.2%})\n"
    else:
        report += "No missing values found.\n"
    report += "\n"

    # Duplicates
    num_duplicates = df.duplicated().sum()
    if num_duplicates > 0:
        report += f"Duplicate Rows: {num_duplicates}\n"
    else:
        report += "No duplicate rows found.\n"
    report += "\n"

    # Data types
    report += "Data Types:\n"
    report += df.dtypes.to_string()
    report += "\n\n"

    # Basic statistics for numerical columns
    numerical_cols = df.select_dtypes(include=np.number).columns
    if not numerical_cols.empty:
        report += "Numerical Column Statistics:\n"
        report += df[numerical_cols].describe().to_string()
        report += "\n\n"
    
    # Unique values for categorical columns
    categorical_cols = df.select_dtypes(include='object').columns
    if not categorical_cols.empty:
        report += "Categorical Column Unique Values (Top 5):\n"
        for col in categorical_cols:
            report += f"  - {col}: {df[col].nunique()} unique values. Examples: {df[col].value_counts().head(5).index.tolist()}\n"
        report += "\n"

    report += "---------------------------\n"
    return report