# tests/test_model_save_load.py
# ═══════════════════════════════════════════════════════════
#  Bio-ML Agent — Model Kaydetme & Yükleme Testleri
#
#  Çalıştır:
#    python -m pytest tests/test_model_save_load.py -v
# ═══════════════════════════════════════════════════════════

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split

from utils.model_compare import ModelComparator
from utils.model_loader import load_model, load_and_predict, model_info, predict_single


# ─────────────────────────────────────────────
#  Fixture'lar
# ─────────────────────────────────────────────

@pytest.fixture
def iris_data():
    """Iris veri seti (küçük, hızlı test için)."""
    data = load_iris()
    X_tr, X_te, y_tr, y_te = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    return X_tr, X_te, y_tr, y_te, data.feature_names


@pytest.fixture
def trained_comparator(iris_data):
    """Eğitilmiş ModelComparator."""
    X_tr, X_te, y_tr, y_te, _ = iris_data
    comp = ModelComparator(task_type="classification", cv_folds=2)
    comp.run(X_tr, X_te, y_tr, y_te)
    return comp


@pytest.fixture
def saved_model_dir(trained_comparator):
    """Modelin kaydedildiği geçici klasör."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trained_comparator.save_results(tmpdir)
        yield tmpdir


# ─────────────────────────────────────────────
#  ModelComparator — save_best_model Testleri
# ─────────────────────────────────────────────

class TestSaveBestModel:
    """save_best_model() testleri."""

    def test_saves_pkl_file(self, trained_comparator):
        """En iyi model .pkl dosyası oluşturulur."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = trained_comparator.save_best_model(tmpdir)
            assert path is not None
            assert path.exists()
            assert path.suffix == ".pkl"

    def test_saves_meta_json(self, trained_comparator):
        """Meta veri JSON dosyası oluşturulur."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = trained_comparator.save_best_model(tmpdir)
            meta_path = path.parent / "best_model_meta.json"
            assert meta_path.exists()

            meta = json.loads(meta_path.read_text())
            assert "model_name" in meta
            assert "task_type" in meta
            assert "metrics" in meta
            assert meta["task_type"] == "classification"

    def test_meta_contains_metrics(self, trained_comparator):
        """Meta veride doğruluk metrikleri bulunur."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trained_comparator.save_best_model(tmpdir)
            meta_path = Path(tmpdir) / "best_model_meta.json"
            meta = json.loads(meta_path.read_text())

            metrics = meta["metrics"]
            assert "accuracy" in metrics
            assert "f1" in metrics
            assert metrics["accuracy"] > 0.5  # Mantıklı bir değer

    def test_no_model_returns_none(self):
        """Model eğitilmemişse None döner."""
        comp = ModelComparator(task_type="classification")
        with tempfile.TemporaryDirectory() as tmpdir:
            result = comp.save_best_model(tmpdir)
            assert result is None

    def test_prefix_works(self, trained_comparator):
        """Dosya adı prefix'i doğru uygulanır."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = trained_comparator.save_best_model(tmpdir, prefix="test")
            assert "test_" in path.name


# ─────────────────────────────────────────────
#  ModelComparator — save_results Testleri
# ─────────────────────────────────────────────

class TestSaveResults:
    """save_results() testleri (joblib entegrasyonu)."""

    def test_includes_model_in_saved(self, trained_comparator):
        """save_results artık model dosyasını da kaydeder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            saved = trained_comparator.save_results(tmpdir)
            assert "model" in saved
            assert saved["model"].exists()

    def test_all_four_files_created(self, trained_comparator):
        """JSON, Markdown, CSV ve Model dosyalarının hepsi oluşturulur."""
        with tempfile.TemporaryDirectory() as tmpdir:
            saved = trained_comparator.save_results(tmpdir)
            assert "json" in saved
            assert "markdown" in saved
            assert "csv" in saved
            assert "model" in saved

    def test_save_model_false_skips_model(self, trained_comparator):
        """save_model=False iken model kaydedilmez."""
        with tempfile.TemporaryDirectory() as tmpdir:
            saved = trained_comparator.save_results(tmpdir, save_model=False)
            assert "model" not in saved


# ─────────────────────────────────────────────
#  ModelComparator — save_all_models Testleri
# ─────────────────────────────────────────────

class TestSaveAllModels:
    """save_all_models() testleri."""

    def test_saves_all_models(self, trained_comparator):
        """Tüm modeller kaydedilir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            saved = trained_comparator.save_all_models(tmpdir)
            assert len(saved) == len(trained_comparator.models)

    def test_models_in_subdirectory(self, trained_comparator):
        """Modeller models/ alt klasöründe oluşturulur."""
        with tempfile.TemporaryDirectory() as tmpdir:
            saved = trained_comparator.save_all_models(tmpdir)
            for path in saved.values():
                assert "models" in str(path)


# ─────────────────────────────────────────────
#  ModelComparator — load_model Testleri
# ─────────────────────────────────────────────

class TestLoadModel:
    """ModelComparator.load_model() testleri."""

    def test_loads_saved_model(self, saved_model_dir):
        """Kaydedilen model başarıyla yüklenir."""
        model_path = str(Path(saved_model_dir) / "best_model.pkl")
        model = ModelComparator.load_model(model_path)
        assert model is not None
        assert hasattr(model, "predict")

    def test_loaded_model_predicts(self, saved_model_dir, iris_data):
        """Yüklenen model tahmin yapabilir."""
        _, X_te, _, y_te, _ = iris_data
        model_path = str(Path(saved_model_dir) / "best_model.pkl")
        model = ModelComparator.load_model(model_path)
        preds = model.predict(X_te)
        assert len(preds) == len(y_te)

    def test_predictions_match(self, saved_model_dir, iris_data):
        """Yüklenen modelin tahminleri makul doğruluktadır."""
        _, X_te, _, y_te, _ = iris_data
        model_path = str(Path(saved_model_dir) / "best_model.pkl")
        model = ModelComparator.load_model(model_path)
        preds = model.predict(X_te)
        accuracy = (preds == y_te).mean()
        assert accuracy > 0.8  # Iris için en az %80

    def test_file_not_found_raises(self):
        """Var olmayan dosya FileNotFoundError fırlatır."""
        with pytest.raises(FileNotFoundError):
            ModelComparator.load_model("/nonexistent/model.pkl")


# ─────────────────────────────────────────────
#  model_loader.py — Standalone Fonksiyon Testleri
# ─────────────────────────────────────────────

class TestModelLoaderFunctions:
    """utils/model_loader.py fonksiyon testleri."""

    def test_load_model_function(self, saved_model_dir):
        """load_model() fonksiyonu çalışır."""
        model_path = str(Path(saved_model_dir) / "best_model.pkl")
        model = load_model(model_path)
        assert model is not None
        assert hasattr(model, "predict")

    def test_load_and_predict(self, saved_model_dir, iris_data):
        """load_and_predict() fonksiyonu doğru çalışır."""
        _, X_te, _, y_te, _ = iris_data
        model_path = str(Path(saved_model_dir) / "best_model.pkl")
        preds = load_and_predict(model_path, X_te)
        assert len(preds) == len(y_te)
        assert isinstance(preds, np.ndarray)

    def test_load_and_predict_proba(self, saved_model_dir, iris_data):
        """load_and_predict(return_proba=True) olasılık döndürür."""
        _, X_te, _, _, _ = iris_data
        model_path = str(Path(saved_model_dir) / "best_model.pkl")
        probas = load_and_predict(model_path, X_te, return_proba=True)
        assert probas.shape[0] == X_te.shape[0]
        assert probas.shape[1] == 3  # Iris: 3 sınıf

    def test_model_info(self, saved_model_dir):
        """model_info() meta bilgileri döndürür."""
        model_path = str(Path(saved_model_dir) / "best_model.pkl")
        info = model_info(model_path)
        assert "model_path" in info
        assert "model_name" in info
        assert "task_type" in info
        assert "metrics" in info

    def test_model_info_pipeline_steps(self, saved_model_dir):
        """model_info() pipeline adımlarını gösterir."""
        model_path = str(Path(saved_model_dir) / "best_model.pkl")
        info = model_info(model_path)
        assert "pipeline_steps" in info
        assert isinstance(info["pipeline_steps"], list)

    def test_predict_single(self, saved_model_dir, iris_data):
        """predict_single() tek örnek tahmini yapar."""
        _, _, _, _, feature_names = iris_data
        model_path = str(Path(saved_model_dir) / "best_model.pkl")

        features = {name: 5.0 for name in feature_names}
        prediction, proba = predict_single(model_path, features, list(feature_names))

        assert prediction is not None
        assert proba is not None
        assert len(proba) == 3  # Iris: 3 sınıf

    def test_load_nonexistent_raises(self):
        """Var olmayan dosya FileNotFoundError fırlatır."""
        with pytest.raises(FileNotFoundError):
            load_model("/nonexistent/path.pkl")


# ─────────────────────────────────────────────
#  Path Strip Testleri (agent.py)
# ─────────────────────────────────────────────

class TestPathStrip:
    """_strip_redundant_prefixes() testleri."""

    def setup_method(self):
        from agent import _strip_redundant_prefixes
        self.strip = _strip_redundant_prefixes

    def test_workspace_prefix_stripped(self):
        """workspace/ prefix'i silinir."""
        assert self.strip("workspace/proj/src/train.py", "proj") == "src/train.py"

    def test_double_nesting_stripped(self):
        """workspace/proj/workspace/proj/ iç içe yapı düzeltilir."""
        result = self.strip("workspace/proj/workspace/proj/src/train.py", "proj")
        assert result == "src/train.py"

    def test_known_root_detected(self):
        """src/, data/, results/ gibi bilinen kökler tespit edilir."""
        assert self.strip("anything/garbage/src/train.py", "proj") == "src/train.py"
        assert self.strip("x/y/z/data/raw/file.csv", "proj") == "data/raw/file.csv"
        assert self.strip("a/b/results/plots/fig.png", "proj") == "results/plots/fig.png"

    def test_known_files_detected(self):
        """Bilinen dosyalar (report.md, README.md) tanınır."""
        assert self.strip("workspace/proj/report.md", "proj") == "report.md"
        assert self.strip("workspace/proj/README.md", "proj") == "README.md"

    def test_no_change_when_clean(self):
        """Zaten temiz yol değiştirilmez."""
        assert self.strip("src/train.py", "proj") == "src/train.py"
        assert self.strip("data/raw/file.csv", "proj") == "data/raw/file.csv"

    def test_different_project_name_stripped(self):
        """LLM'in farklı proje adı kullanması durumu."""
        result = self.strip("workspace/diabetes/src/train.py", "scratch_project")
        assert result == "src/train.py"

    def test_only_project_name_stripped(self):
        """Sadece workspace/ olmadan proje adı prefix var."""
        result = self.strip("proj/src/train.py", "proj")
        assert result == "src/train.py"

    def test_deeply_nested(self):
        """Çok derin iç içe yapı."""
        result = self.strip("scratch/workspace/diabetes/workspace/proj/src/train.py", "scratch")
        assert result == "src/train.py"


# ─────────────────────────────────────────────
#  Regresyon Testi (Eski testlerin bozulmadığını doğrula)
# ─────────────────────────────────────────────

class TestRegressionTask:
    """Regresyon görevi ile model kaydetme testi."""

    def test_regression_save_load(self):
        """Regresyon modeli de kaydedilip yüklenebilir."""
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        X_tr, X_te, y_tr, y_te = train_test_split(
            data.data, data.target, test_size=0.2, random_state=42
        )

        comp = ModelComparator(task_type="regression", cv_folds=2)
        comp.run(X_tr, X_te, y_tr, y_te)

        with tempfile.TemporaryDirectory() as tmpdir:
            saved = comp.save_results(tmpdir)
            assert "model" in saved

            model = ModelComparator.load_model(str(saved["model"]))
            preds = model.predict(X_te)
            assert len(preds) == len(y_te)
