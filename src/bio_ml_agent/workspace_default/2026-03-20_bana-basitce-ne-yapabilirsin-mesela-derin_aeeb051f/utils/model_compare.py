import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def compare_models(X_train, X_test, y_train, y_test, task_type="classification", output_dir="results/"):
    """
    Çeşitli makine öğrenmesi modellerini eğitir, çapraz doğrulama yapar, değerlendirir ve karşılaştırır.

    Args:
        X_train, X_test, y_train, y_test: Eğitim ve test veri setleri.
        task_type (str): 'classification' veya 'regression'.
        output_dir (str): Sonuçların ve en iyi modelin kaydedileceği dizin.

    Returns:
        tuple: (None, results_df) - Bir placeholder ve sonuçları içeren DataFrame.
    """
    os.makedirs(output_dir, exist_ok=True)

    if task_type != "classification":
        raise NotImplementedError("Şu anda sadece sınıflandırma ('classification') görevleri desteklenmektedir.")

    models = {
        'LogisticRegression': LogisticRegression(solver='liblinear', random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier()
    }

    results = []
    
    print(f"Karşılaştırılacak Modeller: {', '.join(models.keys())}")

    for name, model in models.items():
        print(f"\n--- Model Eğitiliyor ve Değerlendiriliyor: {name} ---")
        
        # StandardScaler ve modeli içeren bir pipeline oluştur
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        # 5-katlı çapraz doğrulama yap
        cv_scores = cross_validate(pipeline, X_train, y_train, cv=5, 
                                   scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc'],
                                   n_jobs=-1)
        
        print(f"Çapraz Doğrulama (CV) Sonuçları (Ortalama):")
        print(f"  - CV F1-Skoru: {np.mean(cv_scores['test_f1_weighted']):.4f}")
        print(f"  - CV ROC AUC: {np.mean(cv_scores['test_roc_auc']):.4f}")

        # Modeli tüm eğitim verisiyle eğit
        pipeline.fit(X_train, y_train)

        # Test verisi üzerinde tahmin yap
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

        # Metrikleri hesapla
        metrics = {
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred, average='weighted'),
            'F1 Score': f1_score(y_test, y_pred, average='weighted'),
            'ROC AUC': roc_auc_score(y_test, y_pred_proba),
            'CV F1 Mean': np.mean(cv_scores['test_f1_weighted']),
            'CV ROC AUC Mean': np.mean(cv_scores['test_roc_auc'])
        }
        results.append(metrics)

    # Sonuçları DataFrame'e dönüştür ve F1 skoruna göre sırala
    results_df = pd.DataFrame(results).sort_values(by='F1 Score', ascending=False).reset_index(drop=True)

    # En iyi modeli belirle ve kaydet
    best_model_name = results_df.iloc[0]['Model']
    best_model = models[best_model_name]
    best_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', best_model)
    ])
    best_pipeline.fit(X_train, y_train)
    
    model_save_path = os.path.join(output_dir, "best_model.pkl")
    joblib.dump(best_pipeline, model_save_path)
    print(f"\nEn iyi model ({best_model_name}) '{model_save_path}' adresine kaydedildi.")

    # Sonuçları JSON ve Markdown olarak kaydet
    results_json_path = os.path.join(output_dir, 'comparison_results.json')
    results_df.to_json(results_json_path, orient='records', indent=4)
    
    results_md_path = os.path.join(output_dir, 'comparison_report.md')
    results_df.to_markdown(results_md_path, index=False)
    
    print(f"Karşılaştırma raporları '{output_dir}' dizinine kaydedildi.")

    return None, results_df
