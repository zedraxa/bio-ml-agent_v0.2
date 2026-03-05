# utils/hyperparameter_optimizer.py
# ═══════════════════════════════════════════════════════════
#  Bio-ML Agent — Hiperparametre Optimizasyonu
#
#  Kullanım:
#    from utils.hyperparameter_optimizer import optimize_model
#
#    best_model, best_params, results = optimize_model(
#        X_train, y_train,
#        model_name="RandomForest",
#        task_type="classification"
#    )
# ═══════════════════════════════════════════════════════════

import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  Varsayılan Hiperparametre Arama Alanları
# ─────────────────────────────────────────────

CLASSIFICATION_PARAM_GRIDS: Dict[str, Dict[str, list]] = {
    "LogisticRegression": {
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "clf__solver": ["lbfgs", "liblinear"],
        "clf__max_iter": [1000, 2000],
    },
    "RandomForest": {
        "clf__n_estimators": [50, 100, 200],
        "clf__max_depth": [None, 10, 20, 30],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
    },
    "GradientBoosting": {
        "clf__n_estimators": [50, 100, 200],
        "clf__learning_rate": [0.01, 0.1, 0.2],
        "clf__max_depth": [3, 5, 7],
        "clf__min_samples_split": [2, 5],
    },
    "SVM": {
        "clf__C": [0.1, 1, 10],
        "clf__kernel": ["rbf", "linear"],
        "clf__gamma": ["scale", "auto"],
    },
    "KNN": {
        "clf__n_neighbors": [3, 5, 7, 9, 11],
        "clf__weights": ["uniform", "distance"],
        "clf__metric": ["euclidean", "manhattan"],
    },
    "DecisionTree": {
        "clf__max_depth": [None, 5, 10, 20, 30],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__criterion": ["gini", "entropy"],
    },
}

REGRESSION_PARAM_GRIDS: Dict[str, Dict[str, list]] = {
    "Ridge": {
        "reg__alpha": [0.01, 0.1, 1, 10, 100],
    },
    "RandomForest": {
        "reg__n_estimators": [50, 100, 200],
        "reg__max_depth": [None, 10, 20],
        "reg__min_samples_split": [2, 5, 10],
    },
    "GradientBoosting": {
        "reg__n_estimators": [50, 100, 200],
        "reg__learning_rate": [0.01, 0.1, 0.2],
        "reg__max_depth": [3, 5, 7],
    },
    "SVR": {
        "reg__C": [0.1, 1, 10],
        "reg__kernel": ["rbf", "linear"],
    },
    "KNN": {
        "reg__n_neighbors": [3, 5, 7, 9],
        "reg__weights": ["uniform", "distance"],
    },
    "DecisionTree": {
        "reg__max_depth": [None, 5, 10, 20],
        "reg__min_samples_split": [2, 5, 10],
    },
}

# Model sınıfı eşleştirmesi (Pipeline içinde)
_CLF_MODELS = {
    "LogisticRegression": ("clf", LogisticRegression(max_iter=2000)),
    "RandomForest": ("clf", RandomForestClassifier(random_state=42)),
    "GradientBoosting": ("clf", GradientBoostingClassifier(random_state=42)),
    "SVM": ("clf", SVC(probability=True, random_state=42)),
    "KNN": ("clf", KNeighborsClassifier()),
    "DecisionTree": ("clf", DecisionTreeClassifier(random_state=42)),
}

_REG_MODELS = {
    "Ridge": ("reg", Ridge()),
    "RandomForest": ("reg", RandomForestRegressor(random_state=42)),
    "GradientBoosting": ("reg", GradientBoostingRegressor(random_state=42)),
    "SVR": ("reg", SVR()),
    "KNN": ("reg", KNeighborsRegressor()),
    "DecisionTree": ("reg", DecisionTreeRegressor(random_state=42)),
}


# ─────────────────────────────────────────────
#  Ana Optimizasyon Fonksiyonu
# ─────────────────────────────────────────────

def optimize_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str = "RandomForest",
    task_type: str = "classification",
    param_grid: Optional[Dict[str, list]] = None,
    cv: int = 5,
    method: str = "grid",
    n_iter: int = 20,
    scoring: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[BaseEstimator, Dict[str, Any], Dict[str, Any]]:
    """
    Tek bir modelin hiperparametrelerini optimize eder.

    Args:
        X_train: Eğitim özellik matrisi.
        y_train: Eğitim hedef vektörü.
        model_name: Model adı (ör: "RandomForest", "SVM", "KNN").
        task_type: "classification" veya "regression".
        param_grid: Özel parametre arama alanı (None ise varsayılan kullanılır).
        cv: Çapraz doğrulama kat sayısı.
        method: "grid" (GridSearchCV) veya "random" (RandomizedSearchCV).
        n_iter: RandomizedSearchCV için iterasyon sayısı.
        scoring: Değerlendirme metriği (None ise accuracy/r2 kullanılır).
        verbose: İlerleme yazdırsın mı.

    Returns:
        (en_iyi_model, en_iyi_parametreler, tüm_sonuçlar) tuple'ı.

    Raises:
        ValueError: Bilinmeyen model adı.
    """
    task_type = task_type.lower()

    # Model ve parametre grid'ini belirle
    if task_type == "classification":
        models = _CLF_MODELS
        default_grids = CLASSIFICATION_PARAM_GRIDS
        default_scoring = "accuracy"
    else:
        models = _REG_MODELS
        default_grids = REGRESSION_PARAM_GRIDS
        default_scoring = "r2"

    if model_name not in models:
        available = ", ".join(sorted(models.keys()))
        raise ValueError(f"Bilinmeyen model: {model_name!r}. Desteklenen: {available}")

    # Pipeline oluştur
    step_name, estimator = models[model_name]
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (step_name, clone(estimator)),
    ])

    # Parametre grid'i
    grid = param_grid or default_grids.get(model_name, {})
    scoring = scoring or default_scoring

    if verbose:
        print(f"\n🔧 Hiperparametre Optimizasyonu")
        print(f"   Model: {model_name}")
        print(f"   Yöntem: {'Grid Search' if method == 'grid' else f'Random Search ({n_iter} iter)'}")
        print(f"   CV: {cv}-fold")
        print(f"   Scoring: {scoring}")
        print(f"   Parametre alanı: {len(grid)} parametre")
        print("─" * 50)

    # Arama
    start = time.time()

    if method == "random":
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            random_state=42,
            return_train_score=True,
        )
    else:
        search = GridSearchCV(
            pipeline,
            param_grid=grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=True,
        )

    search.fit(X_train, y_train)
    elapsed = time.time() - start

    # Sonuçları hazırla
    results = {
        "model_name": model_name,
        "task_type": task_type,
        "method": method,
        "cv_folds": cv,
        "scoring": scoring,
        "best_score": float(search.best_score_),
        "best_params": {k: _serialize_value(v) for k, v in search.best_params_.items()},
        "elapsed_seconds": round(elapsed, 2),
        "n_candidates": len(search.cv_results_["mean_test_score"]),
        "top_5": _top_n_results(search, n=5),
    }

    if verbose:
        print(f"\n✅ Optimizasyon tamamlandı ({elapsed:.1f}s)")
        print(f"   🏆 En iyi skor: {search.best_score_:.4f}")
        print(f"   📊 Denenen kombinasyon: {results['n_candidates']}")
        print(f"   🔧 En iyi parametreler:")
        for k, v in search.best_params_.items():
            clean_k = k.split("__", 1)[-1] if "__" in k else k
            print(f"      {clean_k}: {v}")

    return search.best_estimator_, search.best_params_, results


def optimize_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str = "classification",
    cv: int = 3,
    method: str = "random",
    n_iter: int = 10,
    output_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Tüm modellerin hiperparametrelerini optimize eder ve karşılaştırır.

    Args:
        X_train: Eğitim özellik matrisi.
        y_train: Eğitim hedef vektörü.
        task_type: "classification" veya "regression".
        cv: Çapraz doğrulama kat sayısı.
        method: "grid" veya "random".
        n_iter: Random search iterasyon sayısı.
        output_dir: Sonuçları kaydetme klasörü.

    Returns:
        Her model için optimizasyon sonuçlarının listesi (en iyiden en kötüye).
    """
    if task_type == "classification":
        model_names = list(_CLF_MODELS.keys())
    else:
        model_names = list(_REG_MODELS.keys())

    all_results: List[Dict[str, Any]] = []

    print(f"\n🔬 Tüm Modeller için Hiperparametre Optimizasyonu")
    print(f"   Görev: {task_type}")
    print(f"   Modeller: {', '.join(model_names)}")
    print("═" * 60)

    for name in model_names:
        try:
            _, _, result = optimize_model(
                X_train, y_train,
                model_name=name,
                task_type=task_type,
                cv=cv,
                method=method,
                n_iter=n_iter,
            )
            all_results.append(result)
        except Exception as e:
            print(f"  ⚠️ {name} optimizasyonu başarısız: {e}")

    # En iyiye göre sırala
    all_results.sort(key=lambda r: r["best_score"], reverse=True)

    # Özet tablo
    print("\n" + "═" * 60)
    print("📊 Optimizasyon Sonuçları (sıralı)")
    print(f"  {'#':<3} {'Model':<22} {'En İyi Skor':<12} {'Süre':<8}")
    print("  " + "─" * 50)
    for i, r in enumerate(all_results, 1):
        badge = "🏆" if i == 1 else "  "
        print(f"  {i:<3}{badge} {r['model_name']:<20} {r['best_score']:<11.4f} {r['elapsed_seconds']:.1f}s")

    # Kaydet
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "optimization_results.json"
        path.write_text(
            json.dumps(all_results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n💾 Sonuçlar kaydedildi: {path}")

    return all_results


# ─────────────────────────────────────────────
#  Yardımcı Fonksiyonlar
# ─────────────────────────────────────────────

def _serialize_value(v: Any) -> Any:
    """NumPy tiplerini JSON-serileştirilebilir hale getir."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def _top_n_results(search, n: int = 5) -> List[Dict[str, Any]]:
    """Grid/Random search sonuçlarından en iyi N sonucu al."""
    cv_results = search.cv_results_
    indices = np.argsort(cv_results["rank_test_score"])[:n]

    top = []
    for idx in indices:
        entry = {
            "rank": int(cv_results["rank_test_score"][idx]),
            "mean_test_score": float(cv_results["mean_test_score"][idx]),
            "std_test_score": float(cv_results["std_test_score"][idx]),
            "mean_train_score": float(cv_results["mean_train_score"][idx]),
            "params": {
                k: _serialize_value(v)
                for k, v in cv_results["params"][idx].items()
            },
        }
        top.append(entry)

    return top


def get_available_models(task_type: str = "classification") -> List[str]:
    """Optimize edilebilecek model adlarını döndür."""
    if task_type == "classification":
        return sorted(_CLF_MODELS.keys())
    return sorted(_REG_MODELS.keys())


def get_param_grid(model_name: str, task_type: str = "classification") -> Dict[str, list]:
    """Belirli bir model için varsayılan parametre grid'ini döndür."""
    if task_type == "classification":
        return CLASSIFICATION_PARAM_GRIDS.get(model_name, {})
    return REGRESSION_PARAM_GRIDS.get(model_name, {})


# ─────────────────────────────────────────────
#  Standalone CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    import pandas as pd
    from sklearn.model_selection import train_test_split

    parser = argparse.ArgumentParser(description="Hiperparametre Optimizasyonu")
    parser.add_argument("--data", required=True, help="CSV veri seti yolu")
    parser.add_argument("--target", required=True, help="Hedef sütun adı")
    parser.add_argument("--task", default="classification",
                        choices=["classification", "regression"])
    parser.add_argument("--model", default=None, help="Model adı (None=tümü)")
    parser.add_argument("--method", default="random", choices=["grid", "random"])
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--output", default="results", help="Çıktı klasörü")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    X = df.drop(columns=[args.target]).values
    y = df[args.target].values
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    if args.model:
        optimize_model(
            X_train, y_train,
            model_name=args.model,
            task_type=args.task,
            cv=args.cv,
            method=args.method,
            n_iter=args.n_iter,
        )
    else:
        optimize_all_models(
            X_train, y_train,
            task_type=args.task,
            cv=args.cv,
            method=args.method,
            n_iter=args.n_iter,
            output_dir=args.output,
        )
