import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedKFold, KFold
import json
import os
import joblib
import numpy as np # numpy import'unu ekledim

def compare_models(X_train, X_test, y_train, y_test, task_type="classification", output_dir="results/"):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42), # probability=True for ROC AUC
        "KNN": KNeighborsClassifier()
    }

    results = {}
    best_model_name = None
    best_score = -float('inf')

    os.makedirs(output_dir, exist_ok=True)
    
    # Define cross-validation strategy
    if task_type == "classification":
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else: # regression
        cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"Eğitiliyor: {name}")
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        # Cross-validation
        fold_metrics = {metric: [] for metric in ["accuracy", "precision", "recall", "f1_score", "roc_auc"] if task_type == "classification"}
        if task_type == "regression":
             fold_metrics = {metric: [] for metric in ["r2", "mae", "rmse"]}

        for fold, (train_idx, val_idx) in enumerate(cv_strategy.split(X_train, y_train)):
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
            
            pipeline.fit(X_train_fold, y_train_fold)
            
            if task_type == "classification":
                y_pred = pipeline.predict(X_val_fold)
                y_proba = pipeline.predict_proba(X_val_fold)[:, 1] if hasattr(pipeline.named_steps['model'], 'predict_proba') else None

                fold_metrics["accuracy"].append(accuracy_score(y_val_fold, y_pred))
                fold_metrics["precision"].append(precision_score(y_val_fold, y_pred, zero_division=0))
                fold_metrics["recall"].append(recall_score(y_val_fold, y_pred, zero_division=0))
                fold_metrics["f1_score"].append(f1_score(y_val_fold, y_pred, zero_division=0))
                if y_proba is not None:
                    fold_metrics["roc_auc"].append(roc_auc_score(y_val_fold, y_proba))
                else:
                    fold_metrics["roc_auc"].append(np.nan)
            else: # regression
                y_pred = pipeline.predict(X_val_fold)
                fold_metrics["r2"].append(r2_score(y_val_fold, y_pred))
                fold_metrics["mae"].append(mean_absolute_error(y_val_fold, y_pred))
                fold_metrics["rmse"].append(np.sqrt(mean_squared_error(y_val_fold, y_pred)))


        # Calculate mean metrics for cross-validation
        avg_metrics = {metric: np.mean(values) for metric, values in fold_metrics.items()}
        results[name] = avg_metrics
        print(f"{name} - Ortalama CV Metrikleri: {avg_metrics}")

        # Determine best model based on a primary metric (e.g., accuracy for classification)
        if task_type == "classification":
            current_score = avg_metrics.get("accuracy", -float('inf'))
        else: # regression
            current_score = avg_metrics.get("r2", -float('inf')) # R2 as primary for regression

        if current_score > best_score:
            best_score = current_score
            best_model_name = name
            
            # Train the best model on the full training data and save it
            best_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model) # Use the specific model instance that performed best
            ])
            best_pipeline.fit(X_train, y_train)
            joblib.dump(best_pipeline, os.path.join(output_dir, "best_model.pkl"))
            print(f"En iyi model '{best_model_name}' kaydedildi.")

    # Save comparison results to JSON
    with open(os.path.join(output_dir, "comparison_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    # Get the best model
    if best_model_name:
        # Load the best pipeline (already saved above)
        best_pipeline = joblib.load(os.path.join(output_dir, "best_model.pkl"))
        print(f"\nEn iyi model: {best_model_name} (Metrik: {best_score:.4f})")
        return best_pipeline, results
    else:
        print("Model karşılaştırması başarısız oldu.")
        return None, results

# Dummy model_loader
class model_loader:
    @staticmethod
    def load_and_predict(model_path, X_new):
        model = joblib.load(model_path)
        return model.predict(X_new)