import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer
import json
import os
import joblib

def compare_models(X_train, X_test, y_train, y_test, task_type="classification", output_dir="results/"):
    os.makedirs(output_dir, exist_ok=True)
    
    models = {
        'LogisticRegression': LogisticRegression(random_state=42, solver='liblinear'),
        'RandomForestClassifier': RandomForestClassifier(random_state=42),
        'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42),
        'SVC': SVC(random_state=42, probability=True), # probability=True for ROC AUC
        'KNeighborsClassifier': KNeighborsClassifier()
    }

    results = {}
    best_model_name = None
    best_score = -np.inf # For classification, typically ROC AUC or F1-score

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted', zero_division=0),
        'recall': make_scorer(recall_score, average='weighted', zero_division=0),
        'f1_score': make_scorer(f1_score, average='weighted', zero_division=0),
        'roc_auc': make_scorer(roc_auc_score, needs_proba=True, average='weighted')
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"Eğitiliyor ve değerlendiriliyor: {name}")
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        try:
            # Cross-validation
            cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_estimator=True)
            
            # Train on full training data and evaluate on test set for final metrics
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, 'predict_proba') else None

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            roc_auc = roc_auc_score(y_test, y_proba, average='weighted') if y_proba is not None else np.nan

            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'cv_accuracy_mean': cv_results['test_accuracy'].mean(),
                'cv_accuracy_std': cv_results['test_accuracy'].std(),
                'cv_roc_auc_mean': cv_results['test_roc_auc'].mean(),
                'cv_roc_auc_std': cv_results['test_roc_auc'].std(),
                'test_pipeline': pipeline # Store the fitted pipeline for saving
            }

            print(f"{name} - Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")

            # Determine the best model based on ROC AUC
            if roc_auc > best_score:
                best_score = roc_auc
                best_model_name = name

        except Exception as e:
            print(f"Model {name} eğitilirken/değerlendirilirken hata oluştu: {e}")
            results[name] = {'error': str(e)}

    # Generate comparison table
    comparison_df = pd.DataFrame({
        name: {
            'Accuracy': res.get('accuracy', np.nan),
            'Precision': res.get('precision', np.nan),
            'Recall': res.get('recall', np.nan),
            'F1-Score': res.get('f1_score', np.nan),
            'ROC AUC': res.get('roc_auc', np.nan),
            'CV Acc Mean': res.get('cv_accuracy_mean', np.nan),
            'CV ROC AUC Mean': res.get('cv_roc_auc_mean', np.nan)
        } for name, res in results.items() if 'error' not in res
    }).T
    
    print("\nModel Karşılaştırma Sonuçları:")
    print(comparison_df.to_markdown(index=True))

    # Save results to JSON
    json_results = {name: {k: v for k, v in res.items() if k != 'test_pipeline'} for name, res in results.items()}
    with open(os.path.join(output_dir, "comparison_results.json"), "w") as f:
        json.dump(json_results, f, indent=4)
    print(f"Karşılaştırma sonuçları kaydedildi: {os.path.join(output_dir, 'comparison_results.json')}")

    # Save comparison report (Markdown)
    with open(os.path.join(output_dir, "comparison_report.md"), "w") as f:
        f.write("# Model Karşılaştırma Raporu\n\n")
        f.write("Aşağıdaki tabloda farklı modellerin performans metrikleri özetlenmektedir:\n\n")
        f.write(comparison_df.to_markdown(index=True))
        f.write(f"\n\n**En İyi Model:** {best_model_name} (ROC AUC'a göre: {best_score:.4f})\n")
    print(f"Karşılaştırma raporu kaydedildi: {os.path.join(output_dir, 'comparison_report.md')}")

    # Save the best model
    best_model_pipeline = None
    if best_model_name and best_model_name in results:
        best_model_pipeline = results[best_model_name]['test_pipeline']
        model_path = os.path.join(output_dir, "best_model.pkl")
        joblib.dump(best_model_pipeline, model_path)
        print(f"En iyi model ('{best_model_name}') kaydedildi: {model_path}")

    return best_model_pipeline, results