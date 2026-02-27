import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import json
import os

def compare_models(X_train, X_test, y_train, y_test, task_type="classification", output_dir="results/"):
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear'),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Support Vector Machine': SVC(random_state=42, probability=True),
        'K-Nearest Neighbors': KNeighborsClassifier()
    }

    results = []
    best_model_name = None
    best_roc_auc = -1 # Or accuracy/r2 for other tasks
    best_model_pipeline = None

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for name, model in models.items():
        print(f"\n--- Eğitim Modeli: {name} ---")
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        # 5-fold cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        cv_accuracy = []
        cv_precision = []
        cv_recall = []
        cv_f1 = []
        cv_roc_auc = []

        fold_idx = 1
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            pipeline.fit(X_cv_train, y_cv_train)
            y_pred_cv = pipeline.predict(X_cv_val)
            y_pred_proba_cv = pipeline.predict_proba(X_cv_val)[:, 1] if task_type == "classification" else None

            cv_accuracy.append(accuracy_score(y_cv_val, y_pred_cv))
            cv_precision.append(precision_score(y_cv_val, y_pred_cv, zero_division=0))
            cv_recall.append(recall_score(y_cv_val, y_pred_cv, zero_division=0))
            cv_f1.append(f1_score(y_cv_val, y_pred_cv, zero_division=0))
            if y_pred_proba_cv is not None:
                cv_roc_auc.append(roc_auc_score(y_cv_val, y_pred_proba_cv))
            print(f"  Fold {fold_idx} - Acc: {cv_accuracy[-1]:.4f}, ROC AUC: {cv_roc_auc[-1]:.4f}")
            fold_idx += 1


        # Train on full X_train and evaluate on X_test
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1] if task_type == "classification" else None

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None

        current_result = {
            'Model': name,
            'CV_Accuracy_Mean': np.mean(cv_accuracy),
            'CV_Precision_Mean': np.mean(cv_precision),
            'CV_Recall_Mean': np.mean(cv_recall),
            'CV_F1_Mean': np.mean(cv_f1),
            'CV_ROC_AUC_Mean': np.mean(cv_roc_auc) if cv_roc_auc else None,
            'Test_Accuracy': accuracy,
            'Test_Precision': precision,
            'Test_Recall': recall,
            'Test_F1': f1,
            'Test_ROC_AUC': roc_auc
        }
        results.append(current_result)

        if roc_auc is not None and roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model_name = name
            best_model_pipeline = pipeline

    results_df = pd.DataFrame(results)
    
    # Save results to JSON
    with open(os.path.join(output_dir, "comparison_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    # Generate Markdown report
    report_path = os.path.join(output_dir, "comparison_report.md")
    with open(report_path, "w") as f:
        f.write("# Model Karşılaştırma Raporu\n\n")
        f.write("Aşağıdaki tabloda farklı modellerin çapraz doğrulama ve test seti üzerindeki performans metrikleri gösterilmektedir.\n\n")
        f.write(results_df.to_markdown(index=False))
        f.write(f"\n\n**En İyi Model (Test ROC AUC'ye Göre):** {best_model_name} (ROC AUC: {best_roc_auc:.4f})\n")

    print(f"\nModel karşılaştırma sonuçları '{output_dir}comparison_results.json' ve '{output_dir}comparison_report.md' dosyalarına kaydedildi.")

    return best_model_pipeline, results_df