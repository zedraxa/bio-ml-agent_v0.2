import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, mean_absolute_error, make_scorer
)
import joblib
import json

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Regression Models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

def get_model_and_params(model_name, task_type="classification"):
    """Returns a model instance and its default hyperparameter search space."""
    if task_type == "classification":
        if model_name == "LogisticRegression":
            model = LogisticRegression(random_state=42, solver='liblinear')
            params = {
                'model__C': [0.01, 0.1, 1, 10, 100],
                'model__penalty': ['l1', 'l2']
            }
        elif model_name == "RandomForestClassifier":
            model = RandomForestClassifier(random_state=42)
            params = {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [None, 10, 20, 30],
                'model__min_samples_split': [2, 5, 10]
            }
        elif model_name == "GradientBoostingClassifier":
            model = GradientBoostingClassifier(random_state=42)
            params = {
                'model__n_estimators': [50, 100, 200],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__max_depth': [3, 5, 7]
            }
        elif model_name == "SVC":
            model = SVC(probability=True, random_state=42)
            params = {
                'model__C': [0.1, 1, 10, 100],
                'model__kernel': ['linear', 'rbf'],
                'model__gamma': ['scale', 'auto']
            }
        elif model_name == "KNeighborsClassifier":
            model = KNeighborsClassifier()
            params = {
                'model__n_neighbors': [3, 5, 7, 9],
                'model__weights': ['uniform', 'distance']
            }
        else:
            raise ValueError(f"Unknown classification model: {model_name}")
    elif task_type == "regression":
        if model_name == "LinearRegression":
            model = LinearRegression()
            params = {} # No hyperparameters to tune for basic LinearRegression
        elif model_name == "Ridge":
            model = Ridge(random_state=42)
            params = {
                'model__alpha': [0.1, 1.0, 10.0]
            }
        elif model_name == "RandomForestRegressor":
            model = RandomForestRegressor(random_state=42)
            params = {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [None, 10, 20, 30],
                'model__min_samples_split': [2, 5, 10]
            }
        elif model_name == "GradientBoostingRegressor":
            model = GradientBoostingRegressor(random_state=42)
            params = {
                'model__n_estimators': [50, 100, 200],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__max_depth': [3, 5, 7]
            }
        elif model_name == "SVR":
            model = SVR()
            params = {
                'model__C': [0.1, 1, 10],
                'model__kernel': ['linear', 'rbf'],
                'model__gamma': ['scale', 'auto']
            }
        elif model_name == "KNeighborsRegressor":
            model = KNeighborsRegressor()
            params = {
                'model__n_neighbors': [3, 5, 7, 9],
                'model__weights': ['uniform', 'distance']
            }
        else:
            raise ValueError(f"Unknown regression model: {model_name}")
    else:
        raise ValueError("task_type must be 'classification' or 'regression'")

    return model, params

def optimize_model(X_train, y_train, model_name, task_type="classification",
                   method="random", n_iter=10, cv=5, scoring='roc_auc',
                   output_dir="results/", random_state=42):
    """
    Performs hyperparameter optimization for a given model.
    """
    model, param_grid = get_model_and_params(model_name, task_type)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    if task_type == "classification":
        # Make sure scoring is appropriate for classification
        if scoring == 'roc_auc' and not hasattr(model, 'predict_proba'):
             print(f"Uyarı: '{model_name}' için 'roc_auc' skoru kullanılamıyor, çünkü predict_proba metodu yok. Varsayılan olarak 'accuracy' kullanılacak.")
             scoring = 'accuracy'
        scorers = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average='weighted', zero_division=0),
            'recall': make_scorer(recall_score, average='weighted', zero_division=0),
            'f1_score': make_scorer(f1_score, average='weighted', zero_division=0),
            'roc_auc': make_scorer(roc_auc_score, needs_proba=True, multi_class='ovo', average='weighted') if hasattr(model, 'predict_proba') else 'accuracy'
        }
    else: # Regression
        scorers = {
            'r2': make_scorer(r2_score),
            'mae': make_scorer(mean_absolute_error),
            'rmse': make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)))
        }

    if method == "random":
        search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=n_iter,
                                    cv=cv, scoring=scoring, refit=scoring, random_state=random_state, n_jobs=-1, verbose=1)
    elif method == "grid":
        search = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring=scoring,
                              refit=scoring, n_jobs=-1, verbose=1)
    else:
        raise ValueError("Method must be 'random' or 'grid'")

    print(f"{model_name} için hiperparametre optimizasyonu başlatılıyor ({method} search)...")
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_
    cv_results = pd.DataFrame(search.cv_results_).sort_values(by=f'rank_test_{scoring}').head(5)

    print(f"\nEn iyi parametreler: {best_params}")
    print(f"En iyi {scoring} skoru (CV): {search.best_score_:.4f}")

    # Save best model
    model_path = f"{output_dir}{model_name}_optimized_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"Optimize edilmiş model şuraya kaydedildi: {model_path}")

    # Save results
    results_path = f"{output_dir}{model_name}_optimization_results.json"
    results_summary = {
        'model_name': model_name,
        'best_score': search.best_score_,
        'best_params': best_params,
        'cv_results_summary': cv_results.to_dict('records')
    }
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=4)
    print(f"Optimizasyon sonuçları şuraya kaydedildi: {results_path}")

    return best_model, best_params, results_summary