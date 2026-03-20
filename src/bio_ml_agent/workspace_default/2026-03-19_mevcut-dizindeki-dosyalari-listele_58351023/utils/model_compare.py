import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, r2_score,
                             mean_absolute_error, mean_squared_error)

# Sınıflandırma Modelleri
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Regresyon Modelleri
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

class ModelComparator:
    def __init__(self, task_type="classification", output_dir="results/"):
        self.task_type = task_type
        self.output_dir = output_dir
        self.models = self._get_models()
        self.results = []
        self.best_model = None
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_models(self):
        if self.task_type == "classification":
            return {
                "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
                "RandomForest": RandomForestClassifier(random_state=42),
                "GradientBoosting": GradientBoostingClassifier(random_state=42),
                "SVM": SVC(probability=True, random_state=42),
                "KNN": KNeighborsClassifier()
            }
        elif self.task_type == "regression":
            return {
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(random_state=42),
                "RandomForest": RandomForestRegressor(random_state=42),
                "GradientBoosting": GradientBoostingRegressor(random_state=42),
                "SVR": SVR(),
                "KNN": KNeighborsRegressor()
            }
        else:
            raise ValueError("task_type must be 'classification' or 'regression'")

    def _get_scoring(self):
        if self.task_type == "classification":
            return ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
        else:
            return ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']
            
    def compare(self, X_train, X_test, y_train, y_test):
        for name, model in self.models.items():
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            
            print(f"--- Training {name} ---")
            
            # Cross-validation
            cv_results = cross_validate(pipeline, X_train, y_train, cv=5, scoring=self._get_scoring())
            
            # Train final model on full training data
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            result_row = {"Model": name}
            
            # Append CV results
            for metric in cv_results:
                if 'test_' in metric:
                    result_row[f"CV {metric.replace('test_', '')} Mean"] = np.mean(cv_results[metric])
                    result_row[f"CV {metric.replace('test_', '')} Std"] = np.std(cv_results[metric])

            # Append test set results
            if self.task_type == "classification":
                y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
                result_row["Test Accuracy"] = accuracy_score(y_test, y_pred)
                result_row["Test Precision"] = precision_score(y_test, y_pred, average='weighted')
                result_row["Test Recall"] = recall_score(y_test, y_pred, average='weighted')
                result_row["Test F1-Score"] = f1_score(y_test, y_pred, average='weighted')
                if y_proba is not None:
                    result_row["Test ROC AUC"] = roc_auc_score(y_test, y_proba)
            else: # Regression
                result_row["Test R2"] = r2_score(y_test, y_pred)
                result_row["Test MAE"] = mean_absolute_error(y_test, y_pred)
                result_row["Test RMSE"] = np.sqrt(mean_squared_error(y_test, y_pred))

            self.results.append(result_row)
            
        results_df = pd.DataFrame(self.results).sort_values(
            by="Test ROC AUC" if self.task_type == "classification" else "Test R2", 
            ascending=False
        ).reset_index(drop=True)
        
        # Identify and save the best model
        best_model_name = results_df.iloc[0]["Model"]
        print(f"\nBest model identified: {best_model_name}")
        
        best_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', self.models[best_model_name])
        ])
        best_pipeline.fit(X_train, y_train)
        self.best_model = best_pipeline
        
        joblib.dump(self.best_model, os.path.join(self.output_dir, "best_model.pkl"))
        
        # Save reports
        results_df.to_json(os.path.join(self.output_dir, "comparison_results.json"), orient='records', indent=4)
        results_df.to_markdown(os.path.join(self.output_dir, "comparison_report.md"), index=False)
        
        return results_df

def compare_models(X_train, X_test, y_train, y_test, task_type="classification", output_dir="results/"):
    comparator = ModelComparator(task_type=task_type, output_dir=output_dir)
    results_df = comparator.compare(X_train, X_test, y_train, y_test)
    return comparator, results_df