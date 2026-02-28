# tests/test_report_generator.py
# ═══════════════════════════════════════════════════════════
#  Bio-ML Agent — Report Generator Test Suite
# ═══════════════════════════════════════════════════════════

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from report_generator import ReportGenerator


# ─────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def report_dir(tmp_path):
    """Geçici rapor dizini."""
    return tmp_path / "test_project"


@pytest.fixture
def gen(report_dir):
    """Temel ReportGenerator instance'ı."""
    return ReportGenerator(project_dir=str(report_dir))


@pytest.fixture
def full_gen(gen):
    """Tüm bilgileri doldurulmuş ReportGenerator."""
    gen.set_dataset_info(
        name="Breast Cancer",
        samples=569,
        features=30,
        target_column="diagnosis",
        task_type="classification",
        source="sklearn",
    )
    gen.set_train_test_info(
        train_size=455,
        test_size=114,
        split_ratio=0.2,
        random_state=42,
        stratified=True,
    )
    gen.add_model_result(
        "LogisticRegression",
        {"accuracy": 0.9561, "f1": 0.9474, "precision": 0.9500, "recall": 0.9449},
    )
    gen.add_model_result(
        "RandomForest",
        {"accuracy": 0.9737, "f1": 0.9677, "precision": 0.9706, "recall": 0.9649},
        is_best=True,
        hyperparameters={"n_estimators": 200, "max_depth": 10},
    )
    gen.add_model_result(
        "SVM",
        {"accuracy": 0.9649, "f1": 0.9565, "precision": 0.9583, "recall": 0.9548},
    )
    gen.add_plot("results/plots/confusion_matrix.png", "Confusion Matrix", "Sınıflandırma hata matrisi")
    gen.add_plot("results/plots/roc_curve.png", "ROC Curve")
    gen.add_note("Veriler normalize edildi.")
    gen.add_note("5-fold cross-validation kullanıldı.")
    return gen


# ─────────────────────────────────────────────
#  Başlatma (Init) Testleri
# ─────────────────────────────────────────────

class TestReportGeneratorInit:

    def test_init_sets_project_dir(self, gen, report_dir):
        assert gen.project_dir == report_dir

    def test_init_sets_project_name(self, gen):
        assert gen.project_name == "test_project"

    def test_init_empty_collections(self, gen):
        assert gen._model_results == []
        assert gen._plots == []
        assert gen._notes == []
        assert gen._dataset_info == {}
        assert gen._best_model is None


# ─────────────────────────────────────────────
#  Veri Toplama Testleri
# ─────────────────────────────────────────────

class TestDataCollection:

    def test_set_dataset_info(self, gen):
        result = gen.set_dataset_info(name="Iris", samples=150, features=4)
        assert gen._dataset_info["name"] == "Iris"
        assert gen._dataset_info["samples"] == 150
        # Fluent API döndürür
        assert result is gen

    def test_set_train_test_info(self, gen):
        result = gen.set_train_test_info(train_size=120, test_size=30)
        assert gen._train_test_info["train_size"] == 120
        assert result is gen

    def test_add_model_result(self, gen):
        result = gen.add_model_result("RF", {"accuracy": 0.95})
        assert len(gen._model_results) == 1
        assert gen._model_results[0]["model_name"] == "RF"
        assert result is gen

    def test_add_best_model(self, gen):
        gen.add_model_result("LR", {"accuracy": 0.90})
        gen.add_model_result("RF", {"accuracy": 0.95}, is_best=True)
        assert gen._best_model == "RF"

    def test_add_hyperparameters(self, gen):
        gen.add_model_result("RF", {"accuracy": 0.95}, hyperparameters={"n_estimators": 100})
        assert gen._hyperparameters["RF"]["n_estimators"] == 100

    def test_add_plot(self, gen):
        result = gen.add_plot("path.png", "Title", "Desc")
        assert len(gen._plots) == 1
        assert gen._plots[0]["title"] == "Title"
        assert result is gen

    def test_add_note(self, gen):
        result = gen.add_note("Test notu")
        assert len(gen._notes) == 1
        assert gen._notes[0] == "Test notu"
        assert result is gen

    def test_multiple_models(self, gen):
        gen.add_model_result("A", {"f1": 0.9})
        gen.add_model_result("B", {"f1": 0.8})
        gen.add_model_result("C", {"f1": 0.7})
        assert len(gen._model_results) == 3


# ─────────────────────────────────────────────
#  Rapor Üretimi Testleri
# ─────────────────────────────────────────────

class TestReportGeneration:

    def test_generate_creates_file(self, full_gen):
        path = full_gen.generate()
        assert path.exists()
        assert path.suffix == ".md"

    def test_generate_in_results_dir(self, full_gen):
        path = full_gen.generate()
        assert "results" in str(path)

    def test_generate_custom_filename(self, full_gen):
        path = full_gen.generate("custom_report.md")
        assert path.name == "custom_report.md"

    def test_generate_creates_results_dir(self, gen):
        gen.add_model_result("RF", {"accuracy": 0.95}, is_best=True)
        path = gen.generate()
        assert path.parent.exists()
        assert path.parent.name == "results"


class TestReportContent:

    def test_contains_header(self, full_gen):
        path = full_gen.generate()
        content = path.read_text(encoding="utf-8")
        assert "Otomatik ML Raporu" in content

    def test_contains_dataset_info(self, full_gen):
        path = full_gen.generate()
        content = path.read_text(encoding="utf-8")
        assert "Breast Cancer" in content
        assert "569" in content
        assert "30" in content

    def test_contains_train_test_split(self, full_gen):
        path = full_gen.generate()
        content = path.read_text(encoding="utf-8")
        assert "Train/Test" in content
        assert "455" in content
        assert "114" in content

    def test_contains_model_results(self, full_gen):
        path = full_gen.generate()
        content = path.read_text(encoding="utf-8")
        assert "LogisticRegression" in content
        assert "RandomForest" in content
        assert "SVM" in content

    def test_contains_best_model_star(self, full_gen):
        path = full_gen.generate()
        content = path.read_text(encoding="utf-8")
        assert "⭐" in content
        assert "En İyi" in content

    def test_contains_comparison_table(self, full_gen):
        path = full_gen.generate()
        content = path.read_text(encoding="utf-8")
        assert "Karşılaştırma Tablosu" in content
        assert "accuracy" in content

    def test_contains_plots_section(self, full_gen):
        path = full_gen.generate()
        content = path.read_text(encoding="utf-8")
        assert "Confusion Matrix" in content
        assert "ROC Curve" in content

    def test_contains_notes(self, full_gen):
        path = full_gen.generate()
        content = path.read_text(encoding="utf-8")
        assert "normalize" in content
        assert "cross-validation" in content

    def test_contains_conclusion(self, full_gen):
        path = full_gen.generate()
        content = path.read_text(encoding="utf-8")
        assert "Sonuç" in content
        assert "RandomForest" in content

    def test_contains_footer(self, full_gen):
        path = full_gen.generate()
        content = path.read_text(encoding="utf-8")
        assert "Bio-ML Agent" in content
        assert "otomatik" in content.lower()

    def test_contains_hyperparameters(self, full_gen):
        path = full_gen.generate()
        content = path.read_text(encoding="utf-8")
        assert "n_estimators" in content


class TestReportEdgeCases:

    def test_empty_report_no_crash(self, gen):
        """Hiçbir bilgi girilmeden rapor oluşturulabilmeli."""
        path = gen.generate()
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "Otomatik ML Raporu" in content

    def test_single_model_no_comparison_table(self, gen):
        """Tek model varsa karşılaştırma tablosu olmamalı."""
        gen.add_model_result("RF", {"accuracy": 0.95})
        path = gen.generate()
        content = path.read_text(encoding="utf-8")
        assert "Karşılaştırma Tablosu" not in content

    def test_no_dataset_info(self, gen):
        gen.add_model_result("RF", {"accuracy": 0.95})
        path = gen.generate()
        content = path.read_text(encoding="utf-8")
        assert "Veri Seti Bilgileri" not in content

    def test_no_train_test_info(self, gen):
        gen.add_model_result("RF", {"accuracy": 0.95})
        path = gen.generate()
        content = path.read_text(encoding="utf-8")
        assert "Train/Test" not in content


# ─────────────────────────────────────────────
#  JSON Yükleme Testleri
# ─────────────────────────────────────────────

class TestLoadFromJson:

    def test_load_from_json_basic(self, gen, tmp_path):
        data = {
            "dataset": {
                "name": "Iris",
                "samples": 150,
                "features": 4,
                "task_type": "classification",
            },
            "best_model": "RandomForest",
            "results": [
                {"model": "LogisticRegression", "accuracy": 0.93, "f1": 0.92},
                {"model": "RandomForest", "accuracy": 0.97, "f1": 0.96},
            ],
        }
        json_path = tmp_path / "results.json"
        json_path.write_text(json.dumps(data), encoding="utf-8")

        result = gen.load_from_json(str(json_path))
        assert result is gen  # Fluent API
        assert gen._dataset_info["name"] == "Iris"
        assert len(gen._model_results) == 2
        assert gen._best_model == "RandomForest"

    def test_load_from_json_nonexistent(self, gen):
        """Olmayan dosyada hata fırlatmaz, uyarı verir."""
        result = gen.load_from_json("/tmp/nonexistent_12345.json")
        assert result is gen
        assert gen._model_results == []

    def test_load_and_generate(self, gen, tmp_path):
        """JSON yükleyip rapor üretme tam döngüsü."""
        data = {
            "best_model": "SVM",
            "results": [
                {"model": "LR", "accuracy": 0.90},
                {"model": "SVM", "accuracy": 0.95},
            ],
        }
        json_path = tmp_path / "results.json"
        json_path.write_text(json.dumps(data), encoding="utf-8")

        gen.load_from_json(str(json_path))
        path = gen.generate()
        content = path.read_text(encoding="utf-8")
        assert "SVM" in content
        assert "⭐" in content
