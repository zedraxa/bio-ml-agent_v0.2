# tests/test_dataset_catalog.py
# ═══════════════════════════════════════════════════════════
#  Bio-ML Agent — Veri Seti Kataloğu Testleri
#
#  Çalıştır:
#    python -m pytest tests/test_dataset_catalog.py -v
# ═══════════════════════════════════════════════════════════

import pytest

from dataset_catalog import (
    DATASET_CATALOG,
    list_datasets,
    get_dataset_info,
    load_dataset,
    search_datasets,
    get_categories,
    format_catalog_for_prompt,
)


# ─────────────────────────────────────────────
#  Katalog Yapısı Testleri
# ─────────────────────────────────────────────

class TestCatalogStructure:
    """DATASET_CATALOG yapı testleri."""

    def test_catalog_not_empty(self):
        """Katalog boş değildir."""
        assert len(DATASET_CATALOG) > 0

    def test_catalog_has_at_least_10(self):
        """Katalogda en az 10 veri seti var."""
        assert len(DATASET_CATALOG) >= 10

    def test_each_entry_has_name(self):
        """Her veri seti 'name' alanına sahip."""
        for ds_id, info in DATASET_CATALOG.items():
            assert "name" in info, f"{ds_id} eksik: name"

    def test_each_entry_has_type(self):
        """Her veri seti 'type' alanına sahip."""
        for ds_id, info in DATASET_CATALOG.items():
            assert "type" in info, f"{ds_id} eksik: type"

    def test_each_entry_has_task(self):
        """Her veri seti 'task' alanına sahip."""
        for ds_id, info in DATASET_CATALOG.items():
            assert "task" in info, f"{ds_id} eksik: task"

    def test_known_datasets_exist(self):
        """Bilinen veri setleri katalogda var."""
        expected = ["breast_cancer", "diabetes", "iris", "wine", "digits"]
        for name in expected:
            assert name in DATASET_CATALOG, f"{name} katalogda yok"


# ─────────────────────────────────────────────
#  list_datasets Testleri
# ─────────────────────────────────────────────

class TestListDatasets:
    """list_datasets() testleri."""

    def test_list_all(self):
        """Tüm veri setleri listelenir."""
        result = list_datasets()
        assert len(result) >= 10

    def test_filter_by_category(self):
        """Kategoriye göre filtreleme çalışır."""
        medical = list_datasets(category="medical")
        assert len(medical) > 0
        for ds in medical:
            assert ds.get("category") == "medical"

    def test_filter_by_task_type(self):
        """Görev türüne göre filtreleme çalışır."""
        clf = list_datasets(task_type="binary_classification")
        assert len(clf) > 0
        for ds in clf:
            assert ds.get("type") == "binary_classification"

    def test_returns_list_of_dicts(self):
        """Sonuç dict listesi."""
        result = list_datasets()
        assert isinstance(result, list)
        assert isinstance(result[0], dict)


# ─────────────────────────────────────────────
#  get_dataset_info Testleri
# ─────────────────────────────────────────────

class TestGetDatasetInfo:
    """get_dataset_info() testleri."""

    def test_valid_dataset(self):
        """Geçerli veri seti bilgisi döner."""
        info = get_dataset_info("breast_cancer")
        assert info is not None
        assert info["name"] == "Wisconsin Breast Cancer"

    def test_invalid_dataset_returns_none(self):
        """Geçersiz veri seti ismi None döner."""
        info = get_dataset_info("nonexistent_dataset")
        assert info is None

    def test_returns_dict(self):
        """Sonuç dict olarak döner."""
        info = get_dataset_info("iris")
        assert isinstance(info, dict)


# ─────────────────────────────────────────────
#  load_dataset Testleri
# ─────────────────────────────────────────────

class TestLoadDataset:
    """load_dataset() testleri."""

    def test_load_breast_cancer(self):
        """breast_cancer veri seti yüklenir."""
        X, y, features = load_dataset("breast_cancer")
        assert X.shape[0] > 0
        assert X.shape[1] == 30
        assert len(y) == X.shape[0]
        assert len(features) == 30

    def test_load_iris(self):
        """iris veri seti yüklenir."""
        X, y, features = load_dataset("iris")
        assert X.shape == (150, 4)
        assert len(y) == 150

    def test_load_diabetes(self):
        """diabetes veri seti yüklenir."""
        X, y, features = load_dataset("diabetes")
        assert X.shape[0] > 0
        assert len(y) == X.shape[0]

    def test_load_wine(self):
        """wine veri seti yüklenir."""
        X, y, features = load_dataset("wine")
        assert X.shape[0] == 178
        assert X.shape[1] == 13

    def test_load_digits(self):
        """digits veri seti yüklenir."""
        X, y, features = load_dataset("digits")
        assert X.shape[0] == 1797
        assert X.shape[1] == 64

    def test_invalid_dataset_raises(self):
        """Geçersiz veri seti ValueError fırlatır."""
        with pytest.raises(ValueError):
            load_dataset("nonexistent_dataset_xyz")

    def test_load_all_datasets(self):
        """Tüm loadable veri setleri yüklenir ve boyutları uyumludur."""
        for ds_id, info in DATASET_CATALOG.items():
            if info.get("loader"):
                X, y, features = load_dataset(ds_id)
                assert X.shape[0] == info["samples"], f"{ds_id} sample count mismatch"
                assert X.shape[1] == info["features"], f"{ds_id} feature count mismatch"
                assert len(y) == X.shape[0], f"{ds_id} target count mismatch"
                assert len(features) == info["features"], f"{ds_id} feature names mismatch"

    def test_returns_tuple(self):
        """Sonuç (X, y, feature_names) tuple döner."""
        result = load_dataset("iris")
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_feature_names_are_strings(self):
        """Feature names string listesi."""
        _, _, features = load_dataset("iris")
        assert all(isinstance(f, str) for f in features)


# ─────────────────────────────────────────────
#  search_datasets Testleri
# ─────────────────────────────────────────────

class TestSearchDatasets:
    """search_datasets() testleri."""

    def test_search_cancer(self):
        """'cancer' araması breast_cancer bulur."""
        results = search_datasets("cancer")
        assert len(results) > 0
        ids = [r.get("id", r.get("name", "")) for r in results]
        # En az bir sonuç cancer içermeli
        found = any("cancer" in str(r).lower() for r in results)
        assert found

    def test_search_diabetes(self):
        """'diabetes' araması sonuç döner."""
        results = search_datasets("diabetes")
        assert len(results) > 0

    def test_search_empty_query(self):
        """Boş arama tüm veri setlerini döner veya boş döner."""
        results = search_datasets("")
        # Boş sorgu ya hepsini döner ya boş — her iki durum OK
        assert isinstance(results, list)

    def test_search_no_results(self):
        """Eşleşme olmayan arama boş liste döner."""
        results = search_datasets("xyznonexistent999")
        assert results == []


# ─────────────────────────────────────────────
#  get_categories Testleri
# ─────────────────────────────────────────────

class TestGetCategories:
    """get_categories() testleri."""

    def test_returns_list(self):
        """Sonuç liste olarak döner."""
        cats = get_categories()
        assert isinstance(cats, list)

    def test_medical_in_categories(self):
        """'medical' kategori listesinde var."""
        cats = get_categories()
        assert "medical" in cats

    def test_not_empty(self):
        """Kategori listesi boş değil."""
        cats = get_categories()
        assert len(cats) > 0


# ─────────────────────────────────────────────
#  format_catalog_for_prompt Testleri
# ─────────────────────────────────────────────

class TestFormatCatalog:
    """format_catalog_for_prompt() testleri."""

    def test_returns_string(self):
        """Sonuç string olarak döner."""
        result = format_catalog_for_prompt()
        assert isinstance(result, str)

    def test_contains_dataset_names(self):
        """Çıktı veri seti adlarını içerir."""
        result = format_catalog_for_prompt()
        assert "breast_cancer" in result
        assert "iris" in result

    def test_not_empty(self):
        """Çıktı boş değil."""
        result = format_catalog_for_prompt()
        assert len(result) > 50
