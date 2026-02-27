# mlflow_tracker.py
# ═══════════════════════════════════════════════════════════
#  Bio-ML Agent — MLflow Entegrasyonu
#  Deney takibi: parametreleri, metrikleri ve modelleri loglar.
#  MLflow yoksa graceful fallback (JSON dosyasına yazar).
# ═══════════════════════════════════════════════════════════

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("bio_ml_agent")

# MLflow'un yüklü olup olmadığını kontrol et
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class MLTracker:
    """MLflow thin wrapper — deney takibi için.

    MLflow yüklüyse MLflow'a loglar.
    MLflow yoksa JSON dosyasına fallback yapar.

    Kullanım:
        tracker = MLTracker(experiment_name="breast_cancer")

        with tracker.start_run(run_name="LogisticRegression"):
            tracker.log_params({"solver": "liblinear", "C": 1.0})
            tracker.log_metrics({"accuracy": 0.95, "f1": 0.94})
            tracker.log_model(model, "model")

        # veya context manager olmadan:
        tracker.start_run("RandomForest")
        tracker.log_params(...)
        tracker.log_metrics(...)
        tracker.end_run()
    """

    def __init__(
        self,
        experiment_name: str = "default",
        tracking_uri: Optional[str] = None,
        fallback_dir: str | Path = "mlflow_logs",
    ):
        self.experiment_name = experiment_name
        self.fallback_dir = Path(fallback_dir)
        self._active_run: Optional[str] = None
        self._run_data: Dict[str, Any] = {}
        self._all_runs: List[Dict[str, Any]] = []
        self._using_mlflow = MLFLOW_AVAILABLE

        if self._using_mlflow:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            log.info("📊 MLflow aktif | experiment=%s", experiment_name)
        else:
            log.info(
                "📊 MLflow bulunamadı — JSON fallback kullanılacak | dir=%s",
                self.fallback_dir,
            )

    @property
    def is_mlflow_active(self) -> bool:
        """MLflow'un aktif olup olmadığını döndür."""
        return self._using_mlflow

    # ── Run Yönetimi ──

    def start_run(self, run_name: str = "run") -> "MLTracker":
        """Yeni bir deney çalışması başlat.

        Context manager olarak da kullanılabilir:
            with tracker.start_run("model_x"):
                ...
        """
        self._active_run = run_name
        self._run_data = {
            "run_name": run_name,
            "experiment": self.experiment_name,
            "started_at": datetime.now().isoformat(),
            "params": {},
            "metrics": {},
            "artifacts": [],
            "tags": {},
        }

        if self._using_mlflow:
            mlflow.start_run(run_name=run_name)
            log.info("📊 MLflow run başlatıldı: %s", run_name)

        return self

    def end_run(self) -> None:
        """Aktif çalışmayı sonlandır."""
        if self._active_run is None:
            return

        self._run_data["ended_at"] = datetime.now().isoformat()
        self._all_runs.append(self._run_data.copy())

        if self._using_mlflow:
            mlflow.end_run()
            log.info("📊 MLflow run tamamlandı: %s", self._active_run)
        else:
            self._save_fallback()

        self._active_run = None
        self._run_data = {}

    def __enter__(self) -> "MLTracker":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.end_run()
        return False

    # ── Loglama ──

    def log_param(self, key: str, value: Any) -> None:
        """Tek bir parametre logla."""
        self._run_data.setdefault("params", {})[key] = value
        if self._using_mlflow and self._active_run:
            mlflow.log_param(key, value)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Birden fazla parametre logla."""
        self._run_data.setdefault("params", {}).update(params)
        if self._using_mlflow and self._active_run:
            mlflow.log_params(params)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Tek bir metrik logla."""
        self._run_data.setdefault("metrics", {})[key] = value
        if self._using_mlflow and self._active_run:
            mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Birden fazla metrik logla."""
        self._run_data.setdefault("metrics", {}).update(metrics)
        if self._using_mlflow and self._active_run:
            mlflow.log_metrics(metrics, step=step)

    def log_model(self, model: Any, artifact_path: str = "model") -> None:
        """Sklearn modelini logla."""
        self._run_data.setdefault("artifacts", []).append({
            "type": "model",
            "path": artifact_path,
            "model_type": type(model).__name__,
        })
        if self._using_mlflow and self._active_run:
            try:
                mlflow.sklearn.log_model(model, artifact_path)
            except Exception as e:
                log.warning("⚠️ MLflow model log başarısız: %s", e)

    def log_artifact(self, filepath: str | Path) -> None:
        """Dosyayı artifact olarak logla."""
        self._run_data.setdefault("artifacts", []).append({
            "type": "file",
            "path": str(filepath),
        })
        if self._using_mlflow and self._active_run:
            try:
                mlflow.log_artifact(str(filepath))
            except Exception as e:
                log.warning("⚠️ MLflow artifact log başarısız: %s", e)

    def set_tag(self, key: str, value: str) -> None:
        """Tag ekle."""
        self._run_data.setdefault("tags", {})[key] = value
        if self._using_mlflow and self._active_run:
            mlflow.set_tag(key, value)

    # ── Sorgulama ──

    def get_all_runs(self) -> List[Dict[str, Any]]:
        """Tüm tamamlanmış run'ların listesini döndür."""
        return list(self._all_runs)

    def get_best_run(self, metric: str = "accuracy", higher_is_better: bool = True) -> Optional[Dict[str, Any]]:
        """En iyi sonuçlu run'ı döndür.

        Args:
            metric: Karşılaştırma metriği.
            higher_is_better: True ise en yüksek, False ise en düşük değer.
        """
        valid_runs = [
            r for r in self._all_runs
            if metric in r.get("metrics", {})
        ]
        if not valid_runs:
            return None
        return (max if higher_is_better else min)(
            valid_runs,
            key=lambda r: r["metrics"][metric],
        )

    # ── Fallback (JSON) ──

    def _save_fallback(self) -> None:
        """MLflow yoksa JSON dosyasına kaydet."""
        self.fallback_dir.mkdir(parents=True, exist_ok=True)
        run_name = self._run_data.get("run_name", "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.experiment_name}_{run_name}_{timestamp}.json"
        filepath = self.fallback_dir / filename

        filepath.write_text(
            json.dumps(self._run_data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        log.info("📊 Fallback JSON kaydedildi: %s", filepath)

    def save_summary(self, output_path: str | Path = "mlflow_summary.json") -> Path:
        """Tüm run'ların özetini JSON olarak kaydet."""
        path = Path(output_path)
        summary = {
            "experiment": self.experiment_name,
            "total_runs": len(self._all_runs),
            "backend": "mlflow" if self._using_mlflow else "json_fallback",
            "runs": self._all_runs,
        }
        path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return path
