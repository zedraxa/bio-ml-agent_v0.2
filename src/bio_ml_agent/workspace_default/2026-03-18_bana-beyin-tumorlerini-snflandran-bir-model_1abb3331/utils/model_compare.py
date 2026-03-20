import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import numpy as np
import json
import os
import joblib # for model saving

def compare_models(X_train, X_test, y_train, y_test, task_type="classification", output_dir="results/"):
    models = {
        'LogisticRegression': LogisticRegression(random_state=42, solver='liblinear'),
        'RandomForestClassifier': RandomForestClassifier(random_state=42),
        'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42),
        'SVC': SVC(probability=True, random_state=42), # probability=True for ROC AUC
        'KNeighborsClassifier': KNeighborsClassifier()
    }

    results = {}
    best_model = None
    best_score = -1.0
    best_model_name = ""

    os.makedirs(output_dir, exist_ok=True)

    for name, model in models.items():
        print(f"Eğitiliyor: {name}")
        
        # Create a pipeline with StandardScaler and the model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test) if hasattr(pipeline, 'predict_proba') else None

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        roc_auc = "N/A"
        if y_prob is not None:
            if task_type == "classification" and len(np.unique(y_train)) > 2: # Multiclass ROC AUC
                # For multiclass, roc_auc_score requires one-vs-rest strategy
                from sklearn.preprocessing import label_binarize
                y_test_binarized = label_binarize(y_test, classes=np.unique(y_train))
                roc_auc = roc_auc_score(y_test_binarized, y_prob, multi_class='ovr', average='weighted')
            elif task_type == "classification" and len(np.unique(y_train)) == 2: # Binary ROC AUC
                roc_auc = roc_auc_score(y_test, y_prob[:, 1])

        current_results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
        results[name] = current_results

        print(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

        if accuracy > best_score: # Using accuracy as the primary metric for comparison
            best_score = accuracy
            best_model = pipeline
            best_model_name = name

    # Save comparison results
    results_filepath = os.path.join(output_dir, "comparison_results.json")
    with open(results_filepath, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Model comparison results saved to {results_filepath}")

    # Save the best model
    if best_model:
        best_model_filepath = os.path.join(output_dir, "best_model.pkl")
        joblib.dump(best_model, best_model_filepath)
        print(f"Best model ({best_model_name}) saved to {best_model_filepath}")
    
    # Generate a markdown report
    report_filepath = os.path.join(output_dir, "comparison_report.md")
    with open(report_filepath, "w") as f:
        f.write("# Model Karşılaştırma Raporu\n\n")
        f.write("Bu rapor, beyin tümörü sınıflandırma görevi için farklı geleneksel makine öğrenimi modellerinin performansını özetlemektedir.\n\n")
        f.write("## Modeller ve Metrikler\n\n")
        f.write("| Model Adı | Accuracy | Precision | Recall | F1-Score | ROC AUC |\n")
        f.write("|---|---|---|---|---|---|\n")
        for name, res in results.items():
            f.write(f"| {name} | {res['accuracy']:.4f} | {res['precision']:.4f} | {res['recall']:.4f} | {res['f1_score']:.4f} | {res['roc_auc']:.4f if isinstance(res['roc_auc'], float) else res['roc_auc']} |\n")
        f.write(f"\nEn iyi model: **{best_model_name}** (Accuracy: {best_score:.4f})\n")
    print(f"Model karşılaştırma raporu kaydedildi: {report_filepath}")

    return best_model, results