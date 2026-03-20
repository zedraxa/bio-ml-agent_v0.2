import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    r2_score, mean_absolute_error, mean_squared_error
)
import joblib
import json
import os

class ModelComparator:
    def __init__(self, output_dir="results/"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.models = {
            "LogisticRegression": LogisticRegression(random_state=42, solver='liblinear'),
            "RandomForestClassifier": RandomForestClassifier(random_state=42),
            "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
            "SVC": SVC(probability=True, random_state=42),
            "KNeighborsClassifier": KNeighborsClassifier(),
        }
        self.regression_models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(random_state=42),
            "RandomForestRegressor": RandomForestRegressor(random_state=42),
            "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
            "SVR": SVR(),
            "KNeighborsRegressor": KNeighborsRegressor(),
        }
        self.results = {}

    def _get_classification_metrics(self, y_true, y_pred, y_proba=None):
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='weighted'),
            "recall": recall_score(y_true, y_pred, average='weighted'),
            "f1_score": f1_score(y_true, y_pred, average='weighted')
        }
        if y_proba is not None:
            # For binary classification, roc_auc_score expects probabilities of the positive class
            # For multi-class, it expects y_proba as (n_samples, n_classes)
            if y_proba.ndim == 1 or y_proba.shape[1] == 2: # Binary case
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
            else: # Multi-class case
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
        return metrics

    def _get_regression_metrics(self, y_true, y_pred):
        metrics = {
            "r2": r2_score(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred))
        }
        return metrics

    def compare_models(self, X_train, X_test, y_train, y_test, task_type="classification", n_splits=5):
        if task_type == "classification":
            scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc_ovr_weighted'] if len(np.unique(y_train)) > 2 else ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            models_to_use = self.models
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        elif task_type == "regression":
            scoring = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']
            models_to_use = self.regression_models
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            raise ValueError("task_type must be 'classification' or 'regression'")

        best_model_name = None
        best_score = -np.inf # For classification, maximize accuracy/roc_auc. For regression, minimize error.

        for name, model in models_to_use.items():
            print(f"Eğitiliyor: {name}")
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])

            # Perform cross-validation
            cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, return_train_score=False)

            # Train the pipeline on the full training data
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            if task_type == "classification":
                if hasattr(pipeline, 'predict_proba'):
                    y_proba = pipeline.predict_proba(X_test)
                else:
                    y_proba = None
                test_metrics = self._get_classification_metrics(y_test, y_pred, y_proba)
                primary_score_name = 'roc_auc' if 'roc_auc' in test_metrics else 'accuracy' # Use ROC AUC if available
                avg_cv_score = np.mean(cv_results[f'test_{primary_score_name}']) if f'test_{primary_score_name}' in cv_results else np.mean(cv_results['test_accuracy'])
            else: # regression
                test_metrics = self._get_regression_metrics(y_test, y_pred)
                primary_score_name = 'r2'
                avg_cv_score = np.mean(cv_results['test_r2']) # Maximize R2 for regression

            self.results[name] = {
                "cv_scores": {k: np.mean(v) for k, v in cv_results.items() if k.startswith('test_')},
                "test_metrics": test_metrics,
                "model_path": os.path.join(self.output_dir, f"{name}.pkl")
            }

            # Save the trained model
            joblib.dump(pipeline, self.results[name]["model_path"])

            # Determine the best model
            if task_type == "classification":
                current_score = test_metrics[primary_score_name]
                if current_score > best_score:
                    best_score = current_score
                    best_model_name = name
            else: # regression
                # For regression, we typically look for highest R2 or lowest error.
                # Here, let's assume we want to maximize R2.
                current_score = test_metrics[primary_score_name]
                if current_score > best_score:
                    best_score = current_score
                    best_model_name = name

        if best_model_name:
            # Save the best model with a generic name
            joblib.dump(joblib.load(self.results[best_model_name]["model_path"]),
                        os.path.join(self.output_dir, "best_model.pkl"))
            self.results["best_model"] = best_model_name

        self._save_results()
        self._generate_report(task_type, X_train.columns)

        return self, self.results, best_model_name

    def _save_results(self):
        with open(os.path.join(self.output_dir, "comparison_results.json"), "w") as f:
            json.dump(self.results, f, indent=4)

    def _generate_report(self, task_type, feature_names):
        report_path = os.path.join(self.output_dir, "comparison_report.md")
        with open(report_path, "w") as f:
            f.write("# Model Karşılaştırma Raporu\n\n")
            f.write("Bu rapor, çeşitli makine öğrenimi modellerinin performansını karşılaştırmaktadır.\n\n")

            f.write("## Modeller ve Metrikler\n\n")
            if task_type == "classification":
                f.write("| Model Adı | Accuracy (Test) | Precision (Test) | Recall (Test) | F1-Score (Test) | ROC-AUC (Test) | Ortalama CV Accuracy |\n")
                f.write("|---|---|---|---|---|---|---|\n")
                for name, data in self.results.items():
                    if name == "best_model": continue
                    test_metrics = data["test_metrics"]
                    cv_accuracy = data["cv_scores"].get("test_accuracy", "N/A")
                    f.write(
                        f"| {name} | {test_metrics['accuracy']:.4f} | {test_metrics['precision']:.4f} | {test_metrics['recall']:.4f} | {test_metrics['f1_score']:.4f} | {test_metrics.get('roc_auc', 0.0):.4f} | {cv_accuracy:.4f} |\n"
                    )
            elif task_type == "regression":
                f.write("| Model Adı | R2 (Test) | MAE (Test) | RMSE (Test) | Ortalama CV R2 |\n")
                f.write("|---|---|---|---|---|\n")
                for name, data in self.results.items():
                    if name == "best_model": continue
                    test_metrics = data["test_metrics"]
                    cv_r2 = data["cv_scores"].get("test_r2", "N/A")
                    f.write(
                        f"| {name} | {test_metrics['r2']:.4f} | {test_metrics['mae']:.4f} | {test_metrics['rmse']:.4f} | {cv_r2:.4f} |\n"
                    )

            if "best_model" in self.results:
                best_model_name = self.results["best_model"]
                f.write(f"\n## En İyi Model: {best_model_name}\n\n")
                f.write(f"Test Seti Metrikleri:\n")
                best_model_metrics = self.results[best_model_name]["test_metrics"]
                for metric, value in best_model_metrics.items():
                    f.write(f"- {metric.replace('_', ' ').title()}: {value:.4f}\n")
                f.write(f"\nModel dosyası kaydedildi: `{self.output_dir}best_model.pkl`\n")
                f.write("\nEn iyi model, test setindeki performansına göre belirlenmiştir.\n")

            f.write("\n## Kullanım Talimatları\n\n")
            f.write("Eğitilmiş en iyi modeli yüklemek ve yeni verilerle tahmin yapmak için aşağıdaki kod örneğini kullanabilirsiniz:\n\n")
            f.write("```python\n")
            f.write("import joblib\n")
            f.write("import pandas as pd\n")
            f.write("from sklearn.preprocessing import StandardScaler\n\n")
            f.write(f"model_path = '{self.output_dir}best_model.pkl'\n")
            f.write("loaded_pipeline = joblib.load(model_path)\n\n")
            f.write("# Yeni veri (örnek olarak tek bir gözlem)\n")
            f.write("new_data = pd.DataFrame([/* yeni veri buraya */], columns=[")
            f.write(", ".join([f"'{col}'" for col in feature_names]))
            f.write("])\n")
            f.write("# Tahmin yap\n")
            f.write("predictions = loaded_pipeline.predict(new_data)\n")
            if task_type == "classification":
                f.write("probabilities = loaded_pipeline.predict_proba(new_data)\n")
                f.write("print(f'Tahmin edilen sınıf: {predictions[0]}')\n")
                f.write("print(f'Sınıf olasılıkları: {probabilities[0]}')\n")
            else:
                f.write("print(f'Tahmin edilen değer: {predictions[0]}')\n")
            f.write("```\n")

def compare_models(X_train, X_test, y_train, y_test, task_type="classification", output_dir="results/", n_splits=5):
    comparator = ModelComparator(output_dir)
    return comparator.compare_models(X_train, X_test, y_train, y_test, task_type, n_splits)

# Regression models (to avoid import error if only classification is used)
try:
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import KFold
except ImportError:
    pass