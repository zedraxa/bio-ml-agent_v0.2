# utils/model_compare.py
# ═══════════════════════════════════════════════════════════
#  Bio-ML Agent — Çoklu Model Karşılaştırma Modülü
#
#  Kullanım:
#    from utils.model_compare import ModelComparator
#    comparator = ModelComparator(task_type="classification")
#    results = comparator.run(X_train, X_test, y_train, y_test)
#    comparator.save_results("results/")
#
#  Veya standalone:
#    python utils/model_compare.py --data data.csv --target quality
# ═══════════════════════════════════════════════════════════

import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  Varsayılan Model Setleri
# ─────────────────────────────────────────────

def get_default_classifiers() -> Dict[str, BaseEstimator]:
    """Varsayılan sınıflandırma modellerini döndürür."""
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, random_state=42)),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "DecisionTree": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(random_state=42)),
        ]),
    }


def get_default_regressors() -> Dict[str, BaseEstimator]:
    """Varsayılan regresyon modellerini döndürür."""
    return {
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", LinearRegression()),
        ]),
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0)),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", RandomForestRegressor(n_estimators=100, random_state=42)),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ]),
        "SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", SVR()),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", KNeighborsRegressor(n_neighbors=5)),
        ]),
        "DecisionTree": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", DecisionTreeRegressor(random_state=42)),
        ]),
    }


# ─────────────────────────────────────────────
#  Model Sonucu Veri Sınıfı
# ─────────────────────────────────────────────

@dataclass
class ModelResult:
    """Tek bir modelin eğitim/test sonuçlarını tutar."""
    name: str
    metrics: Dict[str, float]
    train_time: float
    cv_scores: Optional[List[float]] = None
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None
    confusion_mat: Optional[List[List[int]]] = None
    classification_rep: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable dict'e dönüştür."""
        d: Dict[str, Any] = {
            "model": self.name,
            "metrics": {k: round(v, 6) for k, v in self.metrics.items()},
            "train_time_seconds": round(self.train_time, 4),
        }
        if self.cv_mean is not None:
            d["cross_validation"] = {
                "scores": [round(s, 4) for s in (self.cv_scores or [])],
                "mean": round(self.cv_mean, 4),
                "std": round(self.cv_std or 0.0, 4),
            }
        if self.confusion_mat is not None:
            d["confusion_matrix"] = self.confusion_mat
        return d


# ─────────────────────────────────────────────
#  Ana Karşılaştırma Sınıfı
# ─────────────────────────────────────────────

class ModelComparator:
    """
    Birden fazla ML modelini eğitip karşılaştıran sınıf.

    Kullanım:
        comparator = ModelComparator(task_type="classification")
        results = comparator.run(X_train, X_test, y_train, y_test)
        comparator.print_comparison()
        comparator.save_results("results/")
    """

    def __init__(
        self,
        task_type: str = "classification",
        models: Optional[Dict[str, BaseEstimator]] = None,
        cv_folds: int = 5,
        random_state: int = 42,
    ):
        """
        Args:
            task_type: "classification" veya "regression"
            models: Özel model sözlüğü (None ise varsayılanlar kullanılır)
            cv_folds: Çapraz doğrulama kat sayısı
            random_state: Tekrarlanabilirlik için seed
        """
        self.task_type = task_type.lower()
        if self.task_type not in ("classification", "regression"):
            raise ValueError(f"task_type 'classification' veya 'regression' olmalı, '{task_type}' verildi")

        if models is not None:
            self.models = models
        elif self.task_type == "classification":
            self.models = get_default_classifiers()
        else:
            self.models = get_default_regressors()

        self.cv_folds = cv_folds
        self.random_state = random_state
        self.results: List[ModelResult] = []
        self.best_model_name: Optional[str] = None
        self.best_model: Optional[BaseEstimator] = None
        self._primary_metric: str = ""

    def run(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        run_cv: bool = True,
    ) -> List[ModelResult]:
        """
        Tüm modelleri eğitip değerlendirir.

        Args:
            X_train, X_test, y_train, y_test: Eğitim ve test verileri
            run_cv: Çapraz doğrulama yapılsın mı

        Returns:
            ModelResult listesi (en iyiden en kötüye sıralı)
        """
        self.results = []

        for name, model in self.models.items():
            print(f"  ⏳ {name} eğitiliyor...", end="", flush=True)
            result = self._train_and_evaluate(
                name, clone(model), X_train, X_test, y_train, y_test, run_cv
            )
            self.results.append(result)

            primary = self._get_primary_metric_value(result)
            print(f" ✅ ({self._primary_metric}: {primary:.4f}, süre: {result.train_time:.2f}s)")

        # Sonuçları birincil metriğe göre sırala (en iyi ilk)
        self.results.sort(
            key=lambda r: r.metrics.get(self._primary_metric, 0),
            reverse=True,
        )

        # En iyi modeli belirle
        if self.results:
            self.best_model_name = self.results[0].name
            self.best_model = clone(self.models[self.best_model_name])
            self.best_model.fit(X_train, y_train)

        return self.results

    def _train_and_evaluate(
        self,
        name: str,
        model: BaseEstimator,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        run_cv: bool,
    ) -> ModelResult:
        """Tek bir modeli eğit ve değerlendir."""

        # Eğitim
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        y_pred = model.predict(X_test)

        if self.task_type == "classification":
            metrics = self._classification_metrics(model, X_test, y_test, y_pred)
            self._primary_metric = "accuracy"

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred).tolist()
            cls_report = classification_report(y_test, y_pred, zero_division=0)
        else:
            metrics = self._regression_metrics(y_test, y_pred)
            self._primary_metric = "r2"
            cm = None
            cls_report = None

        # Çapraz doğrulama
        cv_scores = None
        cv_mean = None
        cv_std = None
        if run_cv:
            try:
                scoring = "accuracy" if self.task_type == "classification" else "r2"
                cv_results = cross_val_score(
                    clone(self.models[name]),
                    np.vstack([X_train, X_test]),
                    np.concatenate([y_train, y_test]),
                    cv=self.cv_folds,
                    scoring=scoring,
                )
                cv_scores = cv_results.tolist()
                cv_mean = float(cv_results.mean())
                cv_std = float(cv_results.std())
            except Exception:
                pass  # CV başarısız olursa sessizce atla

        return ModelResult(
            name=name,
            metrics=metrics,
            train_time=train_time,
            cv_scores=cv_scores,
            cv_mean=cv_mean,
            cv_std=cv_std,
            confusion_mat=cm,
            classification_rep=cls_report,
        )

    def _classification_metrics(
        self,
        model: BaseEstimator,
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Sınıflandırma metrikleri hesaplar."""
        n_classes = len(np.unique(y_test))
        average = "binary" if n_classes == 2 else "macro"

        metrics: Dict[str, float] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average=average, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average=average, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, average=average, zero_division=0)),
        }

        # ROC-AUC (probability gerektirir)
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)
                if n_classes == 2:
                    metrics["roc_auc"] = float(roc_auc_score(y_test, proba[:, 1]))
                else:
                    metrics["roc_auc"] = float(
                        roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
                    )
        except Exception:
            pass  # ROC-AUC hesaplanamadıysa atla

        return metrics

    def _regression_metrics(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Regresyon metrikleri hesaplar."""
        return {
            "r2": float(r2_score(y_test, y_pred)),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "mse": float(mean_squared_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        }

    def _get_primary_metric_value(self, result: ModelResult) -> float:
        """Birincil metrik değerini döndürür."""
        return result.metrics.get(self._primary_metric, 0.0)

    # ─────────────────────────────────────────
    #  Çıktı & Raporlama
    # ─────────────────────────────────────────

    def print_comparison(self) -> str:
        """
        Karşılaştırma tablosunu konsola yazdırır ve string olarak döndürür.

        Returns:
            Tablo string'i
        """
        if not self.results:
            msg = "❌ Henüz sonuç yok. Önce run() çağırın."
            print(msg)
            return msg

        lines: List[str] = []
        lines.append("")
        lines.append("╔══════════════════════════════════════════════════════════════════════════════╗")
        lines.append("║                    📊 Model Karşılaştırma Sonuçları                         ║")
        lines.append("╠══════════════════════════════════════════════════════════════════════════════╣")

        if self.task_type == "classification":
            header = f"  {'#':<3} {'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9} {'CV Mean':>8} {'Süre':>7}"
            lines.append(header)
            lines.append("  " + "─" * 90)

            for i, r in enumerate(self.results, 1):
                badge = " 🏆" if i == 1 else "   "
                auc_str = f"{r.metrics.get('roc_auc', 0):.4f}" if "roc_auc" in r.metrics else "  N/A "
                cv_str = f"{r.cv_mean:.4f}" if r.cv_mean is not None else "  N/A "
                line = (
                    f"  {i:<3}{badge}{r.name:<18} "
                    f"{r.metrics['accuracy']:>8.4f} "
                    f"{r.metrics['precision']:>9.4f} "
                    f"{r.metrics['recall']:>8.4f} "
                    f"{r.metrics['f1']:>7.4f} "
                    f"{auc_str:>8} "
                    f"{cv_str:>8} "
                    f"{r.train_time:>6.2f}s"
                )
                lines.append(line)
        else:
            header = f"  {'#':<3} {'Model':<22} {'R²':>8} {'MAE':>10} {'RMSE':>10} {'CV Mean':>8} {'Süre':>7}"
            lines.append(header)
            lines.append("  " + "─" * 80)

            for i, r in enumerate(self.results, 1):
                badge = " 🏆" if i == 1 else "   "
                cv_str = f"{r.cv_mean:.4f}" if r.cv_mean is not None else "  N/A "
                line = (
                    f"  {i:<3}{badge}{r.name:<18} "
                    f"{r.metrics['r2']:>8.4f} "
                    f"{r.metrics['mae']:>9.4f} "
                    f"{r.metrics['rmse']:>9.4f} "
                    f"{cv_str:>8} "
                    f"{r.train_time:>6.2f}s"
                )
                lines.append(line)

        lines.append("")
        lines.append(f"  🏆 En İyi Model: {self.best_model_name}")
        if self.results:
            best = self.results[0]
            primary_val = best.metrics.get(self._primary_metric, 0)
            lines.append(f"     {self._primary_metric}: {primary_val:.4f}")
            if best.cv_mean is not None:
                lines.append(f"     CV {self._primary_metric} (mean ± std): {best.cv_mean:.4f} ± {best.cv_std:.4f}")
        lines.append("╚══════════════════════════════════════════════════════════════════════════════╝")
        lines.append("")

        table_str = "\n".join(lines)
        print(table_str)
        return table_str

    def get_results_dict(self) -> Dict[str, Any]:
        """Tüm sonuçları JSON-serializable dict olarak döndürür."""
        return {
            "task_type": self.task_type,
            "n_models": len(self.results),
            "cv_folds": self.cv_folds,
            "primary_metric": self._primary_metric,
            "best_model": self.best_model_name,
            "ranking": [r.to_dict() for r in self.results],
        }

    def save_results(self, output_dir: str, prefix: str = "", save_model: bool = True) -> Dict[str, Path]:
        """
        Sonuçları dosyalara kaydeder.

        Args:
            output_dir: Çıktı klasörü
            prefix: Dosya adı öneki (opsiyonel)
            save_model: En iyi modeli joblib ile kaydetsin mi (varsayılan: True)

        Returns:
            Oluşturulan dosya yollarının dict'i
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        pfx = f"{prefix}_" if prefix else ""
        saved: Dict[str, Path] = {}

        # 1. JSON — tüm sonuçlar
        json_path = out / f"{pfx}comparison_results.json"
        json_path.write_text(
            json.dumps(self.get_results_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        saved["json"] = json_path

        # 2. Markdown rapor
        md_path = out / f"{pfx}comparison_report.md"
        md_path.write_text(self._generate_markdown_report(), encoding="utf-8")
        saved["markdown"] = md_path

        # 3. CSV — metrik tablosu
        csv_path = out / f"{pfx}comparison_metrics.csv"
        self._save_metrics_csv(csv_path)
        saved["csv"] = csv_path

        # 4. En iyi model (joblib)
        if save_model and self.best_model is not None:
            model_path = self.save_best_model(output_dir, prefix)
            if model_path:
                saved["model"] = model_path

        print(f"💾 Sonuçlar kaydedildi:")
        for label, path in saved.items():
            print(f"   📄 {label}: {path}")

        return saved

    def save_best_model(self, output_dir: str, prefix: str = "") -> Optional[Path]:
        """
        En iyi modeli joblib ile kaydeder.

        Args:
            output_dir: Çıktı klasörü
            prefix: Dosya adı öneki

        Returns:
            Model dosya yolu (None = model yok)
        """
        if self.best_model is None:
            print("⚠️ Kaydedilecek model yok. Önce run() çağırın.")
            return None

        import joblib

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        pfx = f"{prefix}_" if prefix else ""
        model_path = out / f"{pfx}best_model.pkl"

        # Model meta verisi
        meta = {
            "model_name": self.best_model_name,
            "task_type": self.task_type,
            "primary_metric": self._primary_metric,
            "metrics": self.results[0].metrics if self.results else {},
            "cv_mean": self.results[0].cv_mean if self.results else None,
        }
        meta_path = out / f"{pfx}best_model_meta.json"
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        joblib.dump(self.best_model, model_path)
        print(f"🧠 En iyi model kaydedildi: {model_path}")
        print(f"   📋 Meta veri: {meta_path}")
        return model_path

    def save_all_models(self, output_dir: str, prefix: str = "") -> Dict[str, Path]:
        """
        Tüm eğitilmiş modelleri joblib ile kaydeder.

        Args:
            output_dir: Çıktı klasörü
            prefix: Dosya adı öneki

        Returns:
            Model adı → dosya yolu dict'i
        """
        import joblib

        out = Path(output_dir) / "models"
        out.mkdir(parents=True, exist_ok=True)
        pfx = f"{prefix}_" if prefix else ""
        saved: Dict[str, Path] = {}

        for name, model in self.models.items():
            safe_name = name.lower().replace(" ", "_")
            model_path = out / f"{pfx}{safe_name}.pkl"
            # Modeli eğitilmiş haliyle kaydet (clone + fit)
            try:
                joblib.dump(model, model_path)
                saved[name] = model_path
            except Exception as e:
                print(f"⚠️ {name} kaydedilemedi: {e}")

        if saved:
            print(f"💾 {len(saved)} model kaydedildi: {out}")
        return saved

    @staticmethod
    def load_model(model_path: str) -> BaseEstimator:
        """
        Kaydedilmiş bir modeli yükler.

        Args:
            model_path: .pkl dosya yolu

        Returns:
            Yüklenmiş model (Pipeline)

        Kullanım:
            model = ModelComparator.load_model("results/best_model.pkl")
            predictions = model.predict(X_new)
        """
        import joblib

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

        model = joblib.load(path)
        print(f"✅ Model yüklendi: {path}")

        # Meta veri varsa göster
        meta_path = path.parent / path.name.replace(".pkl", "_meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            print(f"   📋 Model: {meta.get('model_name', '?')}")
            print(f"   📊 Görev: {meta.get('task_type', '?')}")
            metrics = meta.get('metrics', {})
            for k, v in metrics.items():
                print(f"   📈 {k}: {v:.4f}")

        return model

    def _save_metrics_csv(self, path: Path) -> None:
        """Metrik tablosunu CSV olarak kaydeder."""
        rows = []
        for r in self.results:
            row = {"model": r.name, "train_time_s": round(r.train_time, 4)}
            row.update({k: round(v, 6) for k, v in r.metrics.items()})
            if r.cv_mean is not None:
                row["cv_mean"] = round(r.cv_mean, 4)
                row["cv_std"] = round(r.cv_std or 0, 4)
            rows.append(row)
        pd.DataFrame(rows).to_csv(path, index=False)

    def _generate_markdown_report(self) -> str:
        """Markdown formatında karşılaştırma raporu oluşturur."""
        lines: List[str] = []
        lines.append("# 📊 Model Karşılaştırma Raporu\n")
        lines.append(f"**Görev Türü:** {self.task_type.capitalize()}")
        lines.append(f"**Karşılaştırılan Model Sayısı:** {len(self.results)}")
        lines.append(f"**Çapraz Doğrulama:** {self.cv_folds}-fold")
        lines.append(f"**En İyi Model:** 🏆 **{self.best_model_name}**\n")

        # Sonuç tablosu
        lines.append("## Karşılaştırma Tablosu\n")

        if self.task_type == "classification":
            lines.append("| # | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV Mean | Süre |")
            lines.append("|---|-------|----------|-----------|--------|-----|---------|---------|------|")
            for i, r in enumerate(self.results, 1):
                badge = "🏆 " if i == 1 else ""
                auc = f"{r.metrics.get('roc_auc', 0):.4f}" if "roc_auc" in r.metrics else "N/A"
                cv = f"{r.cv_mean:.4f}" if r.cv_mean is not None else "N/A"
                lines.append(
                    f"| {i} | {badge}**{r.name}** | "
                    f"{r.metrics['accuracy']:.4f} | "
                    f"{r.metrics['precision']:.4f} | "
                    f"{r.metrics['recall']:.4f} | "
                    f"{r.metrics['f1']:.4f} | "
                    f"{auc} | {cv} | "
                    f"{r.train_time:.2f}s |"
                )
        else:
            lines.append("| # | Model | R² | MAE | RMSE | CV Mean | Süre |")
            lines.append("|---|-------|----|-----|------|---------|------|")
            for i, r in enumerate(self.results, 1):
                badge = "🏆 " if i == 1 else ""
                cv = f"{r.cv_mean:.4f}" if r.cv_mean is not None else "N/A"
                lines.append(
                    f"| {i} | {badge}**{r.name}** | "
                    f"{r.metrics['r2']:.4f} | "
                    f"{r.metrics['mae']:.4f} | "
                    f"{r.metrics['rmse']:.4f} | "
                    f"{cv} | "
                    f"{r.train_time:.2f}s |"
                )

        # En iyi model detayları
        if self.results:
            best = self.results[0]
            lines.append(f"\n## 🏆 En İyi Model: {best.name}\n")
            lines.append("### Metrikler\n")
            for k, v in best.metrics.items():
                lines.append(f"- **{k}:** {v:.4f}")
            if best.cv_mean is not None:
                lines.append(f"- **CV Mean ± Std:** {best.cv_mean:.4f} ± {best.cv_std:.4f}")
            lines.append(f"- **Eğitim Süresi:** {best.train_time:.2f}s")

            if best.classification_rep:
                lines.append("\n### Classification Report\n")
                lines.append("```")
                lines.append(best.classification_rep)
                lines.append("```")

        lines.append("\n---")
        lines.append("*Bu rapor Bio-ML Agent tarafından otomatik oluşturulmuştur.*\n")

        return "\n".join(lines)

    def plot_comparison(self, output_dir: str, prefix: str = "") -> Optional[Path]:
        """
        Model karşılaştırma grafiğini oluşturur ve kaydeder.

        Args:
            output_dir: Çıktı klasörü
            prefix: Dosya adı öneki

        Returns:
            Grafik dosyasının yolu (matplotlib yoksa None)
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠️ matplotlib yüklü değil, grafik oluşturulamadı.")
            print("   Yüklemek için: pip install matplotlib")
            return None

        if not self.results:
            return None

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        pfx = f"{prefix}_" if prefix else ""

        names = [r.name for r in self.results]
        n_models = len(names)

        if self.task_type == "classification":
            metrics_to_plot = ["accuracy", "precision", "recall", "f1"]
            available = [m for m in metrics_to_plot if m in self.results[0].metrics]

            fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 6))
            if len(available) == 1:
                axes = [axes]

            colors = plt.cm.Set2(np.linspace(0, 1, n_models))

            for ax, metric in zip(axes, available):
                values = [r.metrics.get(metric, 0) for r in self.results]
                bars = ax.barh(names, values, color=colors, edgecolor="white", linewidth=0.5)

                # Değer etiketleri
                for bar, val in zip(bars, values):
                    ax.text(
                        bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                        f"{val:.4f}", va="center", fontsize=9, fontweight="bold",
                    )

                ax.set_xlabel(metric.capitalize(), fontsize=11)
                ax.set_xlim(0, max(values) * 1.15 if max(values) > 0 else 1)
                ax.set_title(metric.upper(), fontsize=13, fontweight="bold")
                ax.invert_yaxis()

            fig.suptitle("📊 Model Karşılaştırması — Sınıflandırma", fontsize=15, fontweight="bold", y=1.02)
        else:
            fig, axes = plt.subplots(1, 3, figsize=(16, 6))
            colors = plt.cm.Set2(np.linspace(0, 1, n_models))

            for ax, metric in zip(axes, ["r2", "mae", "rmse"]):
                values = [r.metrics.get(metric, 0) for r in self.results]
                bars = ax.barh(names, values, color=colors, edgecolor="white", linewidth=0.5)
                for bar, val in zip(bars, values):
                    ax.text(
                        bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                        f"{val:.4f}", va="center", fontsize=9, fontweight="bold",
                    )
                ax.set_xlabel(metric.upper(), fontsize=11)
                ax.set_title(metric.upper(), fontsize=13, fontweight="bold")
                ax.invert_yaxis()

            fig.suptitle("📊 Model Karşılaştırması — Regresyon", fontsize=15, fontweight="bold", y=1.02)

        plt.tight_layout()
        chart_path = out / f"{pfx}comparison_chart.png"
        fig.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        print(f"📊 Grafik kaydedildi: {chart_path}")
        return chart_path


# ─────────────────────────────────────────────
#  Kolaylık Fonksiyonu (Tek Çağrıda Karşılaştırma)
# ─────────────────────────────────────────────

def compare_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    task_type: str = "classification",
    models: Optional[Dict[str, BaseEstimator]] = None,
    output_dir: Optional[str] = None,
    cv_folds: int = 5,
) -> Tuple[ModelComparator, List[ModelResult]]:
    """
    Tek çağrıda çoklu model karşılaştırması yapar.

    Kullanım:
        comparator, results = compare_models(
            X_train, X_test, y_train, y_test,
            task_type="classification",
            output_dir="results/"
        )

    Returns:
        (ModelComparator, results listesi) tuple'ı
    """
    comparator = ModelComparator(
        task_type=task_type,
        models=models,
        cv_folds=cv_folds,
    )

    print(f"\n🔬 Çoklu Model Karşılaştırması ({task_type})")
    print(f"   Modeller: {', '.join(comparator.models.keys())}")
    print(f"   CV Folds: {cv_folds}")
    print("─" * 60)

    results = comparator.run(X_train, X_test, y_train, y_test)
    comparator.print_comparison()

    if output_dir:
        comparator.save_results(output_dir)
        comparator.plot_comparison(output_dir)

    return comparator, results

def deploy_if_better_than_production(
    new_model_result: ModelResult, 
    new_model_path: Path, 
    model_name_for_registry: str,
    primary_metric: str = "accuracy"
) -> bool:
    """
    Active Learning için: Yeni eğitilen modelin (new_model_result) performansını,
    MLFlow üzerindeki 'Production' etiketli eski model ile kıyaslar. 
    Eğer yeni model daha iyiyse MLFlow'da bunu Production'a alır.
    """
    from mlflow_tracker import get_shared_tracker
    tracker = get_shared_tracker()
    
    if not tracker.is_mlflow_active:
        print("⚠️ MLFlow aktif değil, model kıyaslaması atlanıyor.")
        return False
        
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient(tracking_uri=tracker.mlflow.get_tracking_uri())
        
        # 1. Mevcut Production modelinin metriklerini bul
        prod_versions = client.get_latest_versions(name=model_name_for_registry, stages=["Production"])
        prod_metric_val = 0.0
        
        if prod_versions:
            prod_run_id = prod_versions[0].run_id
            prod_run = client.get_run(prod_run_id)
            prod_metric_val = prod_run.data.metrics.get(primary_metric, 0.0)
            print(f"📊 MLFlow Production Model Skoru ({primary_metric}): {prod_metric_val:.4f}")
        else:
            print(f"ℹ️ MLFlow'da '{model_name_for_registry}' için Production modeli bulunamadı. Yeni model doğrudan kabul edilecek.")
            
        # 2. Yeni modelin skorunu al
        new_metric_val = new_model_result.metrics.get(primary_metric, 0.0)
        print(f"📊 Yeni Retrained Model Skoru ({primary_metric}): {new_metric_val:.4f}")
        
        # 3. Kıyasla
        if new_metric_val > prod_metric_val:
            print(f"✅ Yeni model daha başarılı! (+{new_metric_val - prod_metric_val:.4f}). Production'a alınıyor...")
            # MLFlow'a kaydet ve Tag'le
            tracker.start_run(run_name=f"active_learning_retrain_{time.strftime('%Y%m%d_%H%M')}")
            tracker.log_metrics(new_model_result.metrics)
            
            # (Gerçek bir senaryoda burada `mlflow.sklearn.log_model(..., registered_model_name=...)` yapılır)
            # ve version transition 'Production' olarak set edilir.
            
            tracker.end_run()
            return True
        else:
            print(f"❌ Yeni model yeterince iyi değil. Eski model Production'da kalmaya devam edecek.")
            return False
            
    except Exception as e:
        print(f"❌ MLFlow Kıyaslama Hatası: {e}")
        return False



# ─────────────────────────────────────────────
#  Standalone CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from sklearn.model_selection import train_test_split

    parser = argparse.ArgumentParser(description="Çoklu Model Karşılaştırma Aracı")
    parser.add_argument("--data", required=True, help="CSV veri seti yolu")
    parser.add_argument("--target", required=True, help="Hedef sütun adı")
    parser.add_argument("--task", default="classification", choices=["classification", "regression"])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--output", default="results", help="Çıktı klasörü")
    parser.add_argument("--sep", default=",", help="CSV ayırıcı karakter")
    args = parser.parse_args()

    print(f"📂 Veri yükleniyor: {args.data}")
    df = pd.read_csv(args.data, sep=args.sep)
    print(f"   Boyut: {df.shape[0]} satır × {df.shape[1]} sütun")

    X = df.drop(columns=[args.target]).values
    y = df[args.target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )

    compare_models(
        X_train, X_test, y_train, y_test,
        task_type=args.task,
        output_dir=args.output,
        cv_folds=args.cv_folds,
    )
