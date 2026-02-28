import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, r2_score, mean_absolute_error, mean_squared_error
import json
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

class ModelComparator:
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = []
        self.models = {}
        self.best_model_name = None
        self.best_model = None
        self.best_metric_value = -np.inf # For classification, higher is better

    def _get_classification_models(self):
        return {
            "LogisticRegression": LogisticRegression(random_state=42, solver='liblinear'),
            "RandomForest": RandomForestClassifier(random_state=42),
            "GradientBoosting": GradientBoostingClassifier(random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "KNN": KNeighborsClassifier()
        }

    def _get_regression_models(self):
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.svm import SVR
        from sklearn.neighbors import KNeighborsRegressor
        return {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(random_state=42),
            "RandomForest": RandomForestRegressor(random_state=42),
            "GradientBoosting": GradientBoostingRegressor(random_state=42),
            "SVR": SVR(),
            "KNN_Regressor": KNeighborsRegressor()
        }

    def _evaluate_classification(self, model, X, y, X_test, y_test, model_name, cv_results):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0] * len(y_test) # Fallback for SVC without probability=True

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba, average='weighted') if hasattr(model, "predict_proba") else 'N/A'

        metrics = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC AUC": roc_auc,
            "CV_Accuracy_Mean": cv_results['test_accuracy'].mean() if 'test_accuracy' in cv_results else None,
            "CV_Accuracy_Std": cv_results['test_accuracy'].std() if 'test_accuracy' in cv_results else None,
        }
        return metrics

    def _evaluate_regression(self, model, X, y, X_test, y_test, model_name, cv_results):
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        metrics = {
            "R2 Score": r2,
            "MAE": mae,
            "RMSE": rmse,
            "CV_R2_Mean": cv_results['test_r2'].mean() if 'test_r2' in cv_results else None,
            "CV_R2_Std": cv_results['test_r2'].std() if 'test_r2' in cv_results else None,
        }
        return metrics

    def compare_models(self, X_train, X_test, y_train, y_test, task_type="classification", cv_folds=5):
        if task_type == "classification":
            base_models = self._get_classification_models()
            scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc_ovr'] # 'roc_auc' for binary
            eval_func = self._evaluate_classification
            primary_metric = "ROC AUC" if len(np.unique(y_train)) == 2 else "Accuracy" # Use ROC AUC for binary, Accuracy for multi-class
        elif task_type == "regression":
            base_models = self._get_regression_models()
            scoring = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']
            eval_func = self._evaluate_regression
            primary_metric = "R2 Score"
        else:
            raise ValueError("task_type must be 'classification' or 'regression'")

        for name, model in base_models.items():
            print(f"Eğitim: {name}...")
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            pipeline.fit(X_train, y_train)
            self.models[name] = pipeline

            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42) if task_type == "classification" else cv_folds
            
            try:
                # For SVC, roc_auc_score requires decision_function or predict_proba
                if name == "SVM" and task_type == "classification":
                    # Use cross_val_predict with method='predict_proba'
                    if len(np.unique(y_train)) == 2: # Binary classification
                        y_pred_proba_cv = cross_val_predict(pipeline, X_train, y_train, cv=cv, method='predict_proba', n_jobs=-1)
                        cv_accuracy = (y_pred_proba_cv.argmax(axis=1) == y_train).mean()
                        cv_roc_auc = roc_auc_score(y_train, y_pred_proba_cv[:, 1])
                        cv_results = {'test_accuracy': np.array([cv_accuracy]), 'test_roc_auc': np.array([cv_roc_auc])} # Simplified for now
                    else: # Multi-class classification
                        y_pred_proba_cv = cross_val_predict(pipeline, X_train, y_train, cv=cv, method='predict_proba', n_jobs=-1)
                        cv_accuracy = (y_pred_proba_cv.argmax(axis=1) == y_train).mean()
                        # For multi-class, roc_auc_score is more complex. Let's simplify for the agent's needs.
                        cv_results = {'test_accuracy': np.array([cv_accuracy])}
                        
                else:
                    from sklearn.model_selection import cross_validate
                    cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
                    if task_type == "regression":
                        # Convert negative scores to positive for display
                        for score_name in ['test_neg_mean_absolute_error', 'test_neg_mean_squared_error']:
                            if score_name in cv_results:
                                cv_results[score_name.replace('neg_', '')] = -cv_results.pop(score_name)
                    if task_type == "classification" and 'test_roc_auc_ovr' in cv_results:
                        cv_results['test_roc_auc'] = cv_results.pop('test_roc_auc_ovr')


            except ValueError as e:
                print(f"Uyarı: {name} modeli için çapraz doğrulama sırasında bir hata oluştu: {e}. Muhtemelen ROC AUC, y_proba gerektiren bir model için hesaplanamıyor.")
                cv_results = {} # Boş bırak veya uygun bir varsayılan ata

            model_metrics = eval_func(pipeline, X_train, y_train, X_test, y_test, name, cv_results)
            self.results.append({"Model": name, **model_metrics})

            # Check if this is the best model based on the primary metric
            current_metric_value = model_metrics.get(primary_metric)
            if current_metric_value is not None:
                if task_type == "classification" and (current_metric_value > self.best_metric_value or self.best_model is None):
                    self.best_metric_value = current_metric_value
                    self.best_model_name = name
                    self.best_model = pipeline
                elif task_type == "regression" and (current_metric_value > self.best_metric_value or self.best_model is None): # R2 higher is better
                    self.best_metric_value = current_metric_value
                    self.best_model_name = name
                    self.best_model = pipeline
        
        self._save_results()
        self._save_best_model()
        self._create_report(task_type)
        return self, pd.DataFrame(self.results)

    def _save_results(self):
        results_path = os.path.join(self.output_dir, "comparison_results.json")
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"Karşılaştırma sonuçları kaydedildi: {results_path}")

    def _save_best_model(self):
        if self.best_model:
            model_path = os.path.join(self.output_dir, "best_model.pkl")
            joblib.dump(self.best_model, model_path)
            print(f"En iyi model ({self.best_model_name}) kaydedildi: {model_path}")

    def _create_report(self, task_type):
        report_path = os.path.join(self.output_dir, "comparison_report.md")
        with open(report_path, 'w') as f:
            f.write("# Model Karşılaştırma Raporu\n\n")
            f.write("Bu rapor, farklı makine öğrenimi modellerinin performansını özetler.\n\n")
            f.write("## Karşılaştırma Tablosu\n\n")
            f.write(pd.DataFrame(self.results).to_markdown(index=False))
            f.write(f"\n\nEn iyi model: **{self.best_model_name}**")
            f.write(f"\n\nModeller, bir StandardScaler ve ardından modelin kendisinden oluşan bir pipeline kullanılarak eğitildi.")
        print(f"Karşılaştırma raporu oluşturuldu: {report_path}")

def compare_models(X_train, X_test, y_train, y_test, task_type="classification", output_dir="results"):
    comparator = ModelComparator(output_dir=output_dir)
    return comparator.compare_models(X_train, X_test, y_train, y_test, task_type=task_type)