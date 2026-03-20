import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
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
import numpy as np

def compare_models(X_train, X_test, y_train, y_test, task_type="classification", output_dir="results/"):
    """
    Çoklu ML modellerini karşılaştırır, en iyi modeli kaydeder ve sonuçları döndürür.

    Args:
        X_train (pd.DataFrame): Eğitim özellikleri.
        X_test (pd.DataFrame): Test özellikleri.
        y_train (pd.Series): Eğitim hedefleri.
        y_test (pd.Series): Test hedefleri.
        task_type (str): "classification" veya "regression".
        output_dir (str): Sonuçların kaydedileceği dizin.

    Returns:
        tuple: (best_model_pipeline, results_df)
    """
    os.makedirs(output_dir, exist_ok=True)

    if task_type == "classification":
        models = {
            "LogisticRegression": Pipeline([
                ('scaler', StandardScaler()),
                ('model', LogisticRegression(solver='liblinear', random_state=42))
            ]),
            "RandomForestClassifier": Pipeline([
                ('scaler', StandardScaler()),
                ('model', RandomForestClassifier(random_state=42))
            ]),
            "GradientBoostingClassifier": Pipeline([
                ('scaler', StandardScaler()),
                ('model', GradientBoostingClassifier(random_state=42))
            ]),
            "SVC": Pipeline([
                ('scaler', StandardScaler()),
                ('model', SVC(probability=True, random_state=42))
            ]),
            "KNeighborsClassifier": Pipeline([
                ('scaler', StandardScaler()),
                ('model', KNeighborsClassifier())
            ])
        }
        metrics_to_report = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        scorer = "roc_auc"
    elif task_type == "regression":
        # Regresyon modelleri eklenebilir
        raise NotImplementedError("Regresyon modelleri henüz implemente edilmedi.")
    else:
        raise ValueError("task_type 'classification' veya 'regression' olmalıdır.")

    results = {}
    best_model_name = None
    best_score = -np.inf if task_type == "regression" else -1

    for name, pipeline in models.items():
        print(f"Eğitim {name}...")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        model_metrics = {}
        if task_type == "classification":
            y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline.named_steps['model'], 'predict_proba') else [0] * len(y_test)
            
            model_metrics["Accuracy"] = accuracy_score(y_test, y_pred)
            model_metrics["Precision"] = precision_score(y_test, y_pred, zero_division=0)
            model_metrics["Recall"] = recall_score(y_test, y_pred, zero_division=0)
            model_metrics["F1-Score"] = f1_score(y_test, y_pred, zero_division=0)
            model_metrics["ROC-AUC"] = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0
            
            current_score = model_metrics["ROC-AUC"]
            if current_score > best_score:
                best_score = current_score
                best_model_name = name
                joblib.dump(pipeline, os.path.join(output_dir, "best_model.pkl"))

        # 5-fold cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if task_type == "classification" else 5
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scorer)
        model_metrics["CV_Mean_Score"] = np.mean(cv_scores)
        model_metrics["CV_Std_Score"] = np.std(cv_scores)

        results[name] = model_metrics

    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df = results_df[metrics_to_report + ["CV_Mean_Score", "CV_Std_Score"]] # Sıralamayı düzenle
    results_df.index.name = "Model"

    print("\nModel Karşılaştırma Sonuçları:")
    print(results_df.round(4))

    # Sonuçları JSON olarak kaydet
    with open(os.path.join(output_dir, "comparison_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    best_model_pipeline = joblib.load(os.path.join(output_dir, "best_model.pkl"))
    print(f"\nEn iyi model ({scorer}): {best_model_name} (Score: {best_score:.4f})")

    return best_model_pipeline, results_df

if __name__ == '__main__':
    # Örnek kullanım (test amaçlı)
    from sklearn.datasets import load_diabetes
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    
    # Diyabet veri seti regresyon içindir, bu yüzden örnek olarak sınıflandırma için uyarlayalım
    # Basitçe medyanın üzerini 1, altını 0 yapalım
    y_classification = (y > y.median()).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y_classification, test_size=0.2, random_state=42, stratify=y_classification)
    
    # compare_models işlevini çağır
    best_model, results_df = compare_models(X_train, X_test, y_train, y_test, task_type="classification")
    print("\nEn İyi Model:", best_model)
    print("\nSonuç DataFrame:\n", results_df)