import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
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

def compare_models(X_train, X_test, y_train, y_test, task_type="classification", output_dir="results/"):
    """
    Compares multiple machine learning models using cross-validation and evaluates them on a test set.

    Args:
        X_train (pd.DataFrame or np.array): Training features.
        X_test (pd.DataFrame or np.array): Test features.
        y_train (pd.Series or np.array): Training target.
        y_test (pd.Series or np.array): Test target.
        task_type (str): "classification" or "regression".
        output_dir (str): Directory to save results and the best model.

    Returns:
        tuple: A tuple containing (best_model, results_df).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    models = {
        "classification": {
            "LogisticRegression": LogisticRegression(random_state=42, solver='liblinear'),
            "RandomForest": RandomForestClassifier(random_state=42),
            "GradientBoosting": GradientBoostingClassifier(random_state=42),
            "SVC": SVC(random_state=42, probability=True),
            "KNeighbors": KNeighborsClassifier()
        },
        "regression": {
            # Add regression models here if needed
        }
    }

    if task_type not in models:
        raise ValueError(f"Unsupported task_type: {task_type}. Choose from {list(models.keys())}")

    results = []
    best_model_name = None
    best_metric_value = -np.inf if task_type == "classification" else -np.inf # Use ROC AUC for classification

    print(f"\n--- Model Comparison ({task_type}) ---")

    for name, model in models[task_type].items():
        print(f"Training and evaluating: {name}")
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        # Cross-validation
        cv_scores = {}
        if task_type == "classification":
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_accuracy = []
            cv_precision = []
            cv_recall = []
            cv_f1 = []
            cv_roc_auc = []
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

                pipeline.fit(X_train_fold, y_train_fold)
                y_pred_fold = pipeline.predict(X_val_fold)
                y_proba_fold = pipeline.predict_proba(X_val_fold)[:, 1] if hasattr(pipeline, 'predict_proba') else None

                cv_accuracy.append(accuracy_score(y_val_fold, y_pred_fold))
                cv_precision.append(precision_score(y_val_fold, y_pred_fold, zero_division=0))
                cv_recall.append(recall_score(y_val_fold, y_pred_fold, zero_division=0))
                cv_f1.append(f1_score(y_val_fold, y_pred_fold, zero_division=0))
                if y_proba_fold is not None:
                    cv_roc_auc.append(roc_auc_score(y_val_fold, y_proba_fold))
            
            cv_scores = {
                "cv_accuracy_mean": np.mean(cv_accuracy), "cv_accuracy_std": np.std(cv_accuracy),
                "cv_precision_mean": np.mean(cv_precision), "cv_precision_std": np.std(cv_precision),
                "cv_recall_mean": np.mean(cv_recall), "cv_recall_std": np.std(cv_recall),
                "cv_f1_mean": np.mean(cv_f1), "cv_f1_std": np.std(cv_f1),
                "cv_roc_auc_mean": np.mean(cv_roc_auc) if cv_roc_auc else 0, "cv_roc_auc_std": np.std(cv_roc_auc) if cv_roc_auc else 0,
            }
        else: # Regression
            # Add regression CV metrics here
            pass

        # Train on full training data and evaluate on test set
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        model_metrics = {"Model": name}
        if task_type == "classification":
            y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, 'predict_proba') else None
            
            model_metrics.update({
                "Test Accuracy": accuracy_score(y_test, y_pred),
                "Test Precision": precision_score(y_test, y_pred, zero_division=0),
                "Test Recall": recall_score(y_test, y_pred, zero_division=0),
                "Test F1-Score": f1_score(y_test, y_pred, zero_division=0),
                "Test ROC AUC": roc_auc_score(y_test, y_proba) if y_proba is not None else 0,
            })
            model_metrics.update(cv_scores)

            current_metric = model_metrics["Test ROC AUC"]
            if current_metric > best_metric_value:
                best_metric_value = current_metric
                best_model_name = name
                best_model = pipeline # Save the entire pipeline
        else: # Regression
            model_metrics.update({
                "Test R2": r2_score(y_test, y_pred),
                "Test MAE": mean_absolute_error(y_test, y_pred),
                "Test RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            })
            # Add logic for best regression model selection
            pass

        results.append(model_metrics)

    results_df = pd.DataFrame(results).sort_values(by="Test ROC AUC", ascending=False if task_type == "classification" else True)
    
    print("\n--- Comparison Results ---")
    print(results_df.round(3).to_markdown(index=False))

    # Save results
    results_df.to_json(os.path.join(output_dir, "comparison_results.json"), orient="records", indent=4)
    with open(os.path.join(output_dir, "comparison_report.md"), "w") as f:
        f.write("# Model Karşılaştırma Raporu\n\n")
        f.write("Aşağıda, eğitim ve test setlerinde çeşitli modellerin performansını özetleyen bir tablo bulunmaktadır:\n\n")
        f.write(results_df.round(4).to_markdown(index=False))
        f.write(f"\n\n**En İyi Model:** {best_model_name} (Test ROC AUC: {best_metric_value:.4f})\n")

    # Save the best model
    if best_model_name:
        joblib.dump(best_model, os.path.join(output_dir, "best_model.pkl"))
        print(f"\nBest model ({best_model_name}) saved to {os.path.join(output_dir, 'best_model.pkl')}")

    return best_model, results_df