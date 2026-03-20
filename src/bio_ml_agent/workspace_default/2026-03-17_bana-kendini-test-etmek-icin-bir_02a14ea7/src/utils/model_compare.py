import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib # for saving/loading models
import json

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Regression Models (if needed in other projects)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

class ModelComparator:
    def __init__(self, models=None, output_dir="results/", random_state=42):
        self.output_dir = output_dir
        self.random_state = random_state
        self.results = {}
        self.best_model_name = None
        self.best_model = None
        self.model_pipelines = {}

        if models is None:
            # Default classification models with pipelines including StandardScaler
            self.models = {
                "LogisticRegression": Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(random_state=self.random_state, solver='liblinear'))]),
                "RandomForestClassifier": Pipeline([('scaler', StandardScaler()), ('model', RandomForestClassifier(random_state=self.random_state))]),
                "GradientBoostingClassifier": Pipeline([('scaler', StandardScaler()), ('model', GradientBoostingClassifier(random_state=self.random_state))]),
                "SVC": Pipeline([('scaler', StandardScaler()), ('model', SVC(probability=True, random_state=self.random_state))]),
                "KNeighborsClassifier": Pipeline([('scaler', StandardScaler()), ('model', KNeighborsClassifier())])
            }
        else:
            self.models = models

    def _evaluate_classification(self, y_true, y_pred, y_proba=None, model_name=""):
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        if y_proba is not None and y_proba.shape[1] == 2: # Binary classification for roc_auc
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
        elif y_proba is not None and y_proba.shape[1] > 2: # Multi-class roc_auc_ovo
             metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovo', average='weighted')
        else:
            metrics['roc_auc'] = np.nan # Not applicable or insufficient data
            
        return metrics

    def _evaluate_regression(self, y_true, y_pred):
        metrics = {
            'r2_score': r2_score(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        }
        return metrics

    def compare_models(self, X_train, X_test, y_train, y_test, task_type="classification", cv=5):
        self.results = {}
        best_score = -np.inf if task_type == "classification" else -np.inf # Maximize accuracy/R2
        
        # Save feature names if available in X_train
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        else:
            self.feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]


        for name, pipeline in self.models.items():
            print(f"Eğitiliyor: {name}...")
            
            # Cross-validation
            kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state) if task_type == "classification" \
                 else KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            
            fold_metrics = []
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                pipeline.fit(X_fold_train, y_fold_train)
                y_fold_pred = pipeline.predict(X_fold_val)
                y_fold_proba = pipeline.predict_proba(X_fold_val) if hasattr(pipeline.named_steps['model'], 'predict_proba') else None

                if task_type == "classification":
                    fold_metrics.append(self._evaluate_classification(y_fold_val, y_fold_pred, y_fold_proba, name))
                else: # Regression
                    fold_metrics.append(self._evaluate_regression(y_fold_val, y_fold_pred))

            # Average CV metrics
            avg_metrics = {metric: np.mean([f[metric] for f in fold_metrics]) for metric in fold_metrics[0]}
            self.results[name] = avg_metrics
            print(f"{name} Ortalama CV Metrikleri: {avg_metrics}")

            # Train on full training data and evaluate on test set for final model
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test) if hasattr(pipeline.named_steps['model'], 'predict_proba') else None

            if task_type == "classification":
                test_metrics = self._evaluate_classification(y_test, y_pred, y_proba, name)
                self.results[name]['test_metrics'] = test_metrics
                print(f"{name} Test Seti Metrikleri: {test_metrics}")
                
                # For classification, accuracy or roc_auc is often the primary metric
                current_score = test_metrics.get('roc_auc', test_metrics.get('accuracy', -np.inf))
            else: # Regression
                test_metrics = self._evaluate_regression(y_test, y_pred)
                self.results[name]['test_metrics'] = test_metrics
                print(f"{name} Test Seti Metrikleri: {test_metrics}")
                current_score = test_metrics.get('r2_score', -np.inf) # For regression, R2 score

            # Update best model based on test set performance
            if current_score > best_score:
                best_score = current_score
                self.best_model_name = name
                self.best_model = pipeline
                
            self.model_pipelines[name] = pipeline # Store all trained pipelines

        print(f"\nEn iyi model: {self.best_model_name} (Skor: {best_score:.4f})")
        self._save_results()
        return self.best_model, self.results

    def _save_results(self):
        if self.best_model:
            model_path = f"{self.output_dir}best_model.pkl"
            joblib.dump(self.best_model, model_path)
            print(f"En iyi model '{self.best_model_name}' şuraya kaydedildi: {model_path}")
        
        results_path = f"{self.output_dir}comparison_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"Karşılaştırma sonuçları şuraya kaydedildi: {results_path}")

        # Generate a markdown report
        report_path = f"{self.output_dir}comparison_report.md"
        with open(report_path, 'w') as f:
            f.write("# Model Karşılaştırma Raporu\n\n")
            f.write("## Özet\n")
            f.write(f"En iyi performans gösteren model: **{self.best_model_name}**\n")
            if self.best_model_name:
                best_model_metrics = self.results[self.best_model_name]['test_metrics']
                f.write("### En İyi Modelin Test Seti Metrikleri:\n")
                for metric, value in best_model_metrics.items():
                    f.write(f"- {metric}: {value:.4f}\n")
            
            f.write("\n## Karşılaştırma Tablosu (Test Seti Metrikleri)\n")
            f.write("| Model Adı | Accuracy | Precision | Recall | F1-Score | ROC AUC |\n")
            f.write("|---|---|---|---|---|---|\n")
            for name, metrics in self.results.items():
                test_m = metrics.get('test_metrics', {})
                f.write(f"| {name} | {test_m.get('accuracy', np.nan):.4f} | {test_m.get('precision', np.nan):.4f} | {test_m.get('recall', np.nan):.4f} | {test_m.get('f1_score', np.nan):.4f} | {test_m.get('roc_auc', np.nan):.4f} |\n")
            f.write("\n")

def load_and_predict(model_path, X_new):
    """Loads a trained model and makes predictions."""
    model = joblib.load(model_path)
    return model.predict(X_new)

# Helper function to get the actual model from a pipeline
def get_model_from_pipeline(pipeline):
    return pipeline.named_steps['model']