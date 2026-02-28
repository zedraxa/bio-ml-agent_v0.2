import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import joblib
import os
import json

class HyperparameterOptimizer:
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.best_model = None
        self.best_params = None
        self.cv_results = None

    def _get_model_and_params(self, model_name, task_type):
        if task_type == "classification":
            if model_name == "LogisticRegression":
                model = LogisticRegression(random_state=42, solver='liblinear')
                params = {
                    'model__C': [0.01, 0.1, 1, 10, 100],
                    'model__penalty': ['l1', 'l2']
                }
            elif model_name == "RandomForest":
                model = RandomForestClassifier(random_state=42)
                params = {
                    'model__n_estimators': [50, 100, 200],
                    'model__max_depth': [None, 10, 20, 30],
                    'model__min_samples_split': [2, 5, 10]
                }
            elif model_name == "GradientBoosting":
                model = GradientBoostingClassifier(random_state=42)
                params = {
                    'model__n_estimators': [50, 100, 200],
                    'model__learning_rate': [0.01, 0.1, 0.2],
                    'model__max_depth': [3, 5, 8]
                }
            elif model_name == "SVM":
                model = SVC(probability=True, random_state=42)
                params = {
                    'model__C': [0.1, 1, 10],
                    'model__kernel': ['linear', 'rbf']
                }
            elif model_name == "KNN":
                model = KNeighborsClassifier()
                params = {
                    'model__n_neighbors': [3, 5, 7, 9],
                    'model__weights': ['uniform', 'distance']
                }
            else:
                raise ValueError(f"Bilinmeyen sınıflandırma modeli: {model_name}")
            scoring = 'roc_auc' if len(np.unique(self.y_train)) == 2 else 'accuracy'

        elif task_type == "regression":
            if model_name == "LinearRegression":
                model = LinearRegression()
                params = {} # Linear Regression usually doesn't need much tuning
            elif model_name == "Ridge":
                model = Ridge(random_state=42)
                params = {'model__alpha': [0.1, 1.0, 10.0]}
            elif model_name == "RandomForest":
                model = RandomForestRegressor(random_state=42)
                params = {
                    'model__n_estimators': [50, 100, 200],
                    'model__max_depth': [None, 10, 20, 30],
                    'model__min_samples_split': [2, 5, 10]
                }
            elif model_name == "GradientBoosting":
                model = GradientBoostingRegressor(random_state=42)
                params = {
                    'model__n_estimators': [50, 100, 200],
                    'model__learning_rate': [0.01, 0.1, 0.2],
                    'model__max_depth': [3, 5, 8]
                }
            elif model_name == "SVR":
                model = SVR()
                params = {
                    'model__C': [0.1, 1, 10],
                    'model__kernel': ['linear', 'rbf'],
                    'model__gamma': ['scale', 'auto']
                }
            elif model_name == "KNN_Regressor":
                model = KNeighborsRegressor()
                params = {
                    'model__n_neighbors': [3, 5, 7, 9],
                    'model__weights': ['uniform', 'distance']
                }
            else:
                raise ValueError(f"Bilinmeyen regresyon modeli: {model_name}")
            scoring = 'r2'
        else:
            raise ValueError("task_type 'classification' veya 'regression' olmalıdır.")

        return model, params, scoring

    def optimize_model(self, X_train, y_train, model_name, task_type="classification", method="grid", n_iter=10, cv=5):
        self.X_train = X_train
        self.y_train = y_train

        base_model, param_grid, scoring = self._get_model_and_params(model_name, task_type)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', base_model)
        ])

        if method == "grid":
            print(f"GridSearchCV ile {model_name} modeli için hiperparametre optimizasyonu başlatılıyor...")
            optimizer = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=1)
        elif method == "random":
            print(f"RandomizedSearchCV ile {model_name} modeli için hiperparametre optimizasyonu başlatılıyor...")
            optimizer = RandomizedSearchCV(pipeline, param_grid, n_iter=n_iter, cv=cv, scoring=scoring, n_jobs=-1, verbose=1, random_state=42)
        else:
            raise ValueError("method 'grid' veya 'random' olmalıdır.")

        optimizer.fit(X_train, y_train)

        self.best_model = optimizer.best_estimator_
        self.best_params = optimizer.best_params_
        self.cv_results = pd.DataFrame(optimizer.cv_results_)

        self._save_optimization_results(model_name)
        
        print(f"\n{model_name} için En İyi Parametreler: {self.best_params}")
        print(f"{model_name} için En İyi Skor: {optimizer.best_score_:.4f}")

        return self.best_model, self.best_params, self.cv_results

    def _save_optimization_results(self, model_name):
        # Save best model
        model_path = os.path.join(self.output_dir, f"best_optimized_{model_name.lower()}.pkl")
        joblib.dump(self.best_model, model_path)
        print(f"En iyi optimize edilmiş model kaydedildi: {model_path}")

        # Save best parameters
        params_path = os.path.join(self.output_dir, f"best_params_{model_name.lower()}.json")
        with open(params_path, 'w') as f:
            json.dump(self.best_params, f, indent=4)
        print(f"En iyi parametreler kaydedildi: {params_path}")

        # Save CV results
        if self.cv_results is not None:
            cv_results_path = os.path.join(self.output_dir, f"cv_results_{model_name.lower()}.csv")
            self.cv_results.to_csv(cv_results_path, index=False)
            print(f"CV sonuçları kaydedildi: {cv_results_path}")

def optimize_model(X_train, y_train, model_name, task_type="classification", method="random", n_iter=10, cv=5, output_dir="results"):
    optimizer = HyperparameterOptimizer(output_dir=output_dir)
    return optimizer.optimize_model(X_train, y_train, model_name, task_type, method, n_iter, cv)