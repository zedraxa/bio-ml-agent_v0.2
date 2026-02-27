# dataset_catalog.py
# ═══════════════════════════════════════════════════════════
#  Bio-ML Agent — Veri Seti Katalog Sistemi
#  Sık kullanılan ML ve biyomühendislik veri setleri.
# ═══════════════════════════════════════════════════════════

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────
#  Veri Seti Kataloğu
# ─────────────────────────────────────────────

DATASET_CATALOG: Dict[str, Dict[str, Any]] = {
    # ── Sklearn İç Veri Setleri ──
    "breast_cancer": {
        "name": "Wisconsin Breast Cancer",
        "loader": "sklearn.datasets.load_breast_cancer",
        "type": "binary_classification",
        "task": "Meme kanseri teşhisi (benign/malign)",
        "features": 30,
        "samples": 569,
        "category": "medical",
        "source": "UCI / sklearn",
    },
    "diabetes": {
        "name": "Diabetes Regression",
        "loader": "sklearn.datasets.load_diabetes",
        "type": "regression",
        "task": "Diyabet hastalığı ilerleme tahmini",
        "features": 10,
        "samples": 442,
        "category": "medical",
        "source": "sklearn",
    },
    "iris": {
        "name": "Iris Flower",
        "loader": "sklearn.datasets.load_iris",
        "type": "multi_classification",
        "task": "Çiçek türü sınıflandırma (3 sınıf)",
        "features": 4,
        "samples": 150,
        "category": "general",
        "source": "sklearn",
    },
    "wine": {
        "name": "Wine Recognition",
        "loader": "sklearn.datasets.load_wine",
        "type": "multi_classification",
        "task": "Şarap türü sınıflandırma (3 sınıf)",
        "features": 13,
        "samples": 178,
        "category": "general",
        "source": "sklearn",
    },
    "digits": {
        "name": "Handwritten Digits",
        "loader": "sklearn.datasets.load_digits",
        "type": "multi_classification",
        "task": "El yazısı rakam tanıma (0-9)",
        "features": 64,
        "samples": 1797,
        "category": "image",
        "source": "sklearn",
    },

    # ── Biyomühendislik Veri Setleri ──
    "heart_disease": {
        "name": "Heart Disease (Cleveland)",
        "url": "https://archive.ics.uci.edu/dataset/45/heart+disease",
        "type": "binary_classification",
        "task": "Kalp hastalığı teşhisi",
        "features": 13,
        "samples": 303,
        "category": "medical",
        "source": "UCI",
        "columns": [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak", "slope",
            "ca", "thal", "target",
        ],
    },
    "parkinsons": {
        "name": "Parkinson's Disease",
        "url": "https://archive.ics.uci.edu/dataset/174/parkinsons",
        "type": "binary_classification",
        "task": "Parkinson hastalığı teşhisi (ses analizi)",
        "features": 22,
        "samples": 195,
        "category": "medical",
        "source": "UCI",
    },
    "liver_disease": {
        "name": "Indian Liver Patient",
        "url": "https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset",
        "type": "binary_classification",
        "task": "Karaciğer hastalığı teşhisi",
        "features": 10,
        "samples": 583,
        "category": "medical",
        "source": "UCI",
    },
    "chronic_kidney": {
        "name": "Chronic Kidney Disease",
        "url": "https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease",
        "type": "binary_classification",
        "task": "Kronik böbrek hastalığı teşhisi",
        "features": 24,
        "samples": 400,
        "category": "medical",
        "source": "UCI",
    },

    # ── Çevre / Biyomühendislik ──
    "water_quality": {
        "name": "Water Quality (Potability)",
        "url": "https://www.kaggle.com/datasets/adityakadiwal/water-potability",
        "type": "binary_classification",
        "task": "İçme suyu kalitesi tahmini",
        "features": 9,
        "samples": 3276,
        "category": "environmental",
        "source": "Kaggle",
        "columns": [
            "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
            "Conductivity", "Organic_carbon", "Trihalomethanes",
            "Turbidity", "Potability",
        ],
    },
    "air_quality": {
        "name": "Air Quality (UCI)",
        "url": "https://archive.ics.uci.edu/dataset/360/air+quality",
        "type": "regression",
        "task": "Hava kalitesi tahmini (sensör verileri)",
        "features": 13,
        "samples": 9358,
        "category": "environmental",
        "source": "UCI",
    },

    # ── Genomik / Biyoinformatik ──
    "gene_expression": {
        "name": "Gene Expression Cancer RNA-Seq",
        "url": "https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq",
        "type": "multi_classification",
        "task": "Kanser türü sınıflandırma (5 tür, RNA-Seq)",
        "features": 20531,
        "samples": 801,
        "category": "genomics",
        "source": "UCI",
    },

    # ── Biyomühendislik Ek Veri Setleri ──
    "eeg_motor_imagery": {
        "name": "EEG Motor Movement/Imagery",
        "url": "https://physionet.org/content/eegmmidb/1.0.0/",
        "type": "multi_classification",
        "task": "Motor hareket/imgeleme sınıflandırma (EEG)",
        "features": 64,
        "samples": 1500,
        "category": "biosignal",
        "source": "PhysioNet",
    },
    "wastewater_treatment": {
        "name": "Water Treatment Plant",
        "url": "https://archive.ics.uci.edu/dataset/63/water+treatment+plant",
        "type": "multi_classification",
        "task": "Atık su arıtma tesisi performans tahmini",
        "features": 38,
        "samples": 527,
        "category": "environmental",
        "source": "UCI",
    },
    "protein_localization": {
        "name": "Yeast Protein Localization",
        "url": "https://archive.ics.uci.edu/dataset/110/yeast",
        "type": "multi_classification",
        "task": "Protein hücresel lokalizasyon tahmini (10 sınıf)",
        "features": 8,
        "samples": 1484,
        "category": "genomics",
        "source": "UCI",
    },
    "molecular_biodegradability": {
        "name": "QSAR Biodegradation",
        "url": "https://archive.ics.uci.edu/dataset/254/qsar+biodegradation",
        "type": "binary_classification",
        "task": "Kimyasal bileşik biyolojik parçalanabilirlik tahmini",
        "features": 41,
        "samples": 1055,
        "category": "drug_discovery",
        "source": "UCI",
    },
    "chest_xray_pneumonia": {
        "name": "Chest X-Ray (Pneumonia)",
        "url": "https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia",
        "type": "binary_classification",
        "task": "Göğüs X-Ray pnömoni tespiti",
        "features": 50176,
        "samples": 5863,
        "category": "medical_imaging",
        "source": "Kaggle",
    },
    "emg_hand_gestures": {
        "name": "EMG Hand Gesture Recognition",
        "url": "https://archive.ics.uci.edu/dataset/481/emg+data+for+gestures",
        "type": "multi_classification",
        "task": "EMG tabanlı el hareketi tanıma",
        "features": 8,
        "samples": 10000,
        "category": "biosignal",
        "source": "UCI",
    },
}


# ─────────────────────────────────────────────
#  Katalog Fonksiyonları
# ─────────────────────────────────────────────

def list_datasets(
    category: Optional[str] = None,
    task_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Katalogdaki veri setlerini listele.

    Args:
        category: Filtreleme kategorisi
                  ("medical", "environmental", "genomics", "general", "image")
        task_type: Görev türü filtresi
                   ("binary_classification", "multi_classification", "regression")

    Returns:
        Veri seti bilgilerinin listesi.
    """
    results = []
    for key, info in DATASET_CATALOG.items():
        if category and info.get("category") != category:
            continue
        if task_type and info.get("type") != task_type:
            continue
        results.append({"id": key, **info})
    return results


def get_dataset_info(dataset_id: str) -> Optional[Dict[str, Any]]:
    """Belirli bir veri setinin bilgilerini getir.

    Args:
        dataset_id: Veri seti kimliği (ör: "breast_cancer", "heart_disease")

    Returns:
        Veri seti bilgileri veya None.
    """
    info = DATASET_CATALOG.get(dataset_id)
    if info is None:
        return None
    return {"id": dataset_id, **info}


def load_dataset(dataset_id: str):
    """Sklearn veri setini yükle (sadece loader tanımlı olanlar).

    Args:
        dataset_id: Veri seti kimliği.

    Returns:
        (X, y, feature_names) tuple.

    Raises:
        ValueError: Veri seti bulunamadı veya yüklenemez.
    """
    info = DATASET_CATALOG.get(dataset_id)
    if info is None:
        available = ", ".join(sorted(DATASET_CATALOG.keys()))
        raise ValueError(
            f"Bilinmeyen veri seti: {dataset_id!r}. "
            f"Mevcut veri setleri: {available}"
        )

    loader_path = info.get("loader")
    if not loader_path:
        raise ValueError(
            f"'{dataset_id}' veri seti otomatik yüklenemez "
            f"(URL tabanlı). URL: {info.get('url', 'N/A')}"
        )

    # Dinamik import: "sklearn.datasets.load_breast_cancer"
    parts = loader_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Geçersiz loader yolu: {loader_path}")

    module_path, func_name = parts
    import importlib
    module = importlib.import_module(module_path)
    loader_func = getattr(module, func_name)

    data = loader_func()
    return data.data, data.target, list(data.feature_names)


def search_datasets(query: str) -> List[Dict[str, Any]]:
    """Veri setlerini anahtar kelimeyle ara.

    Args:
        query: Arama sorgusu.

    Returns:
        Eşleşen veri setleri.
    """
    query_lower = query.lower()
    results = []
    for key, info in DATASET_CATALOG.items():
        searchable = f"{key} {info.get('name', '')} {info.get('task', '')} {info.get('category', '')}".lower()
        if query_lower in searchable:
            results.append({"id": key, **info})
    return results


def get_categories() -> List[str]:
    """Mevcut kategorileri döndür."""
    return sorted(set(info.get("category", "") for info in DATASET_CATALOG.values()))


def format_catalog_for_prompt() -> str:
    """System prompt'a eklenecek katalog özeti oluştur."""
    lines = [
        "\n\nVERİ SETİ KATALOĞU:",
        "Aşağıdaki veri setleri doğrudan kullanılabilir:\n",
    ]
    for key, info in DATASET_CATALOG.items():
        loader = "✅ auto-load" if info.get("loader") else "📥 URL"
        lines.append(
            f"  • {key}: {info['name']} | {info['type']} | "
            f"{info['features']} özellik, {info['samples']} örnek | {loader}"
        )
    lines.append(
        "\nKullanım: `from dataset_catalog import load_dataset; "
        "X, y, names = load_dataset('breast_cancer')`"
    )
    return "\n".join(lines)
