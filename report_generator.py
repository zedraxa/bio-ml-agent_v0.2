# report_generator.py
# ═══════════════════════════════════════════════════════════
#  Bio-ML Agent — Otomatik Rapor Oluşturucu
#  ML projeleri için kapsamlı Markdown rapor üretir.
# ═══════════════════════════════════════════════════════════

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("bio_ml_agent")


class ReportGenerator:
    """ML projesi için otomatik Markdown rapor oluşturur.

    Kullanım:
        gen = ReportGenerator(project_dir="workspace/my_project")
        gen.set_dataset_info(name="Breast Cancer", samples=569, features=30)
        gen.add_model_result("LogisticRegression", {"accuracy": 0.95, "f1": 0.94})
        gen.add_model_result("RandomForest", {"accuracy": 0.97, "f1": 0.96})
        gen.add_plot("results/plots/confusion_matrix.png", "Confusion Matrix")
        report_path = gen.generate()
    """

    def __init__(self, project_dir: str | Path):
        self.project_dir = Path(project_dir)
        self.project_name = self.project_dir.name
        self._dataset_info: Dict[str, Any] = {}
        self._model_results: List[Dict[str, Any]] = []
        self._plots: List[Dict[str, str]] = []
        self._notes: List[str] = []
        self._best_model: Optional[str] = None
        self._train_test_info: Dict[str, Any] = {}
        self._hyperparameters: Dict[str, Dict[str, Any]] = {}

    # ── Veri Toplama ──

    def set_dataset_info(
        self,
        name: str,
        samples: int = 0,
        features: int = 0,
        target_column: str = "",
        task_type: str = "classification",
        source: str = "",
        **extra,
    ) -> "ReportGenerator":
        """Veri seti bilgilerini ayarla."""
        self._dataset_info = {
            "name": name,
            "samples": samples,
            "features": features,
            "target_column": target_column,
            "task_type": task_type,
            "source": source,
            **extra,
        }
        return self

    def set_train_test_info(
        self,
        train_size: int = 0,
        test_size: int = 0,
        split_ratio: float = 0.2,
        random_state: int = 42,
        stratified: bool = True,
    ) -> "ReportGenerator":
        """Train/test split bilgilerini ayarla."""
        self._train_test_info = {
            "train_size": train_size,
            "test_size": test_size,
            "split_ratio": split_ratio,
            "random_state": random_state,
            "stratified": stratified,
        }
        return self

    def add_model_result(
        self,
        model_name: str,
        metrics: Dict[str, float],
        is_best: bool = False,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> "ReportGenerator":
        """Bir model sonucunu ekle.

        Args:
            model_name: Model adı (ör: "LogisticRegression")
            metrics: Metrik sözlüğü (ör: {"accuracy": 0.95, "f1": 0.94})
            is_best: En iyi model mi?
            hyperparameters: Hiperparametreler.
        """
        self._model_results.append({
            "model_name": model_name,
            "metrics": metrics,
            "is_best": is_best,
        })
        if is_best:
            self._best_model = model_name
        if hyperparameters:
            self._hyperparameters[model_name] = hyperparameters
        return self

    def add_plot(self, path: str, title: str, description: str = "") -> "ReportGenerator":
        """Grafik referansı ekle."""
        self._plots.append({
            "path": path,
            "title": title,
            "description": description,
        })
        return self

    def add_note(self, note: str) -> "ReportGenerator":
        """Rapora not ekle."""
        self._notes.append(note)
        return self

    # ── Rapor Üretimi ──

    def generate(self, output_filename: str = "auto_report.md") -> Path:
        """Markdown raporu oluştur ve kaydet.

        Returns:
            Rapor dosyasının yolu.
        """
        sections = [
            self._header(),
            self._dataset_section(),
            self._train_test_section(),
            self._models_section(),
            self._comparison_table(),
            self._plots_section(),
            self._notes_section(),
            self._conclusion(),
            self._footer(),
        ]

        content = "\n\n".join(s for s in sections if s)

        results_dir = self.project_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / output_filename
        output_path.write_text(content, encoding="utf-8")

        log.info("📄 Rapor oluşturuldu: %s", output_path)
        return output_path

    def load_from_json(self, json_path: str | Path) -> "ReportGenerator":
        """comparison_results.json dosyasından sonuçları yükle.

        Args:
            json_path: JSON dosya yolu.
        """
        path = Path(json_path)
        if not path.exists():
            log.warning("⚠️ JSON dosyası bulunamadı: %s", path)
            return self

        data = json.loads(path.read_text(encoding="utf-8"))

        # Genel bilgiler
        if "dataset" in data:
            self.set_dataset_info(**data["dataset"])

        # Model sonuçları
        best_model = data.get("best_model", "")
        for model_data in data.get("results", []):
            name = model_data.get("model", model_data.get("model_name", "Unknown"))
            metrics = {
                k: v for k, v in model_data.items()
                if k not in ("model", "model_name", "hyperparameters")
                and isinstance(v, (int, float))
            }
            self.add_model_result(
                name, metrics,
                is_best=(name == best_model),
                hyperparameters=model_data.get("hyperparameters"),
            )

        return self

    # ── Section Generators ──

    def _header(self) -> str:
        title = self.project_name.replace("_", " ").title()
        date = datetime.now().strftime("%Y-%m-%d %H:%M")
        return (
            f"# 📊 {title} — Otomatik ML Raporu\n\n"
            f"> 🕐 Oluşturulma: {date}  \n"
            f"> 🤖 Bio-ML Agent tarafından otomatik üretildi."
        )

    def _dataset_section(self) -> str:
        if not self._dataset_info:
            return ""
        d = self._dataset_info
        lines = [
            "## 📋 Veri Seti Bilgileri\n",
            f"| Özellik | Değer |",
            f"|---------|-------|",
            f"| **İsim** | {d.get('name', 'N/A')} |",
            f"| **Kaynak** | {d.get('source', 'N/A')} |",
            f"| **Görev Türü** | {d.get('task_type', 'N/A')} |",
            f"| **Örnek Sayısı** | {d.get('samples', 'N/A'):,} |",
            f"| **Özellik Sayısı** | {d.get('features', 'N/A')} |",
        ]
        if d.get("target_column"):
            lines.append(f"| **Hedef Sütun** | `{d['target_column']}` |")
        return "\n".join(lines)

    def _train_test_section(self) -> str:
        if not self._train_test_info:
            return ""
        t = self._train_test_info
        return (
            "## 🔀 Train/Test Split\n\n"
            f"| Parametre | Değer |\n"
            f"|-----------|-------|\n"
            f"| **Train Boyutu** | {t.get('train_size', 'N/A')} |\n"
            f"| **Test Boyutu** | {t.get('test_size', 'N/A')} |\n"
            f"| **Split Oranı** | {t.get('split_ratio', 0.2):.0%} |\n"
            f"| **Random State** | {t.get('random_state', 42)} |\n"
            f"| **Stratified** | {'Evet ✅' if t.get('stratified') else 'Hayır'} |"
        )

    def _models_section(self) -> str:
        if not self._model_results:
            return ""
        lines = ["## 🤖 Model Sonuçları\n"]
        for result in self._model_results:
            name = result["model_name"]
            star = " ⭐ (En İyi)" if result.get("is_best") else ""
            lines.append(f"### {name}{star}\n")

            # Metrikler
            metrics = result.get("metrics", {})
            if metrics:
                lines.append("| Metrik | Değer |")
                lines.append("|--------|-------|")
                for k, v in sorted(metrics.items()):
                    fmt = f"{v:.4f}" if isinstance(v, float) else str(v)
                    lines.append(f"| **{k}** | {fmt} |")
                lines.append("")

            # Hiperparametreler
            hp = self._hyperparameters.get(name, {})
            if hp:
                lines.append(f"**Hiperparametreler:** `{hp}`\n")

        return "\n".join(lines)

    def _comparison_table(self) -> str:
        if len(self._model_results) < 2:
            return ""

        # Tüm metrikleri topla
        all_metrics = set()
        for r in self._model_results:
            all_metrics.update(r.get("metrics", {}).keys())
        all_metrics = sorted(all_metrics)

        if not all_metrics:
            return ""

        lines = ["## 📊 Model Karşılaştırma Tablosu\n"]
        header = "| Model | " + " | ".join(f"**{m}**" for m in all_metrics) + " |"
        sep = "|-------|" + "|".join("------:" for _ in all_metrics) + "|"
        lines.append(header)
        lines.append(sep)

        for result in self._model_results:
            name = result["model_name"]
            star = " ⭐" if result.get("is_best") else ""
            metrics = result.get("metrics", {})
            values = []
            for m in all_metrics:
                v = metrics.get(m)
                if isinstance(v, float):
                    values.append(f"{v:.4f}")
                elif v is not None:
                    values.append(str(v))
                else:
                    values.append("—")
            row = f"| **{name}**{star} | " + " | ".join(values) + " |"
            lines.append(row)

        return "\n".join(lines)

    def _plots_section(self) -> str:
        if not self._plots:
            return ""
        lines = ["## 📈 Grafikler\n"]
        for plot in self._plots:
            lines.append(f"### {plot['title']}")
            if plot.get("description"):
                lines.append(f"\n{plot['description']}\n")
            lines.append(f"\n![{plot['title']}]({plot['path']})\n")
        return "\n".join(lines)

    def _notes_section(self) -> str:
        if not self._notes:
            return ""
        lines = ["## 📝 Notlar\n"]
        for note in self._notes:
            lines.append(f"- {note}")
        return "\n".join(lines)

    def _conclusion(self) -> str:
        if not self._model_results:
            return ""
        lines = ["## 🎯 Sonuç\n"]
        if self._best_model:
            best = next(
                (r for r in self._model_results if r.get("is_best")), None
            )
            if best:
                metrics = best.get("metrics", {})
                top_metric = max(metrics.items(), key=lambda x: x[1]) if metrics else ("", 0)
                lines.append(
                    f"En iyi performans gösteren model **{self._best_model}** olarak belirlenmiştir. "
                    f"Bu model {top_metric[0]} metriğinde **{top_metric[1]:.4f}** değerine ulaşmıştır."
                )
        else:
            lines.append("Model karşılaştırması yapılmıştır. Sonuçlar yukarıdaki tabloda özetlenmiştir.")

        return "\n".join(lines)

    def _footer(self) -> str:
        return (
            "---\n\n"
            "*Bu rapor Bio-ML Agent tarafından otomatik olarak oluşturulmuştur.* 🤖"
        )
