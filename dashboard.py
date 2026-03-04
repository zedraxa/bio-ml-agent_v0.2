# dashboard.py
# ═══════════════════════════════════════════════════════════
#  Bio-ML Agent — Task Dashboard Sunucusu
#  Çalıştırma: python dashboard.py
#  Tarayıcı:   http://localhost:5050
# ═══════════════════════════════════════════════════════════

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from flask import Flask, jsonify, request, send_from_directory, abort

# ─────────────────────────────────────────────
#  Yapılandırma
# ─────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
TASKS_FILE = BASE_DIR / "tasks.json"
REPORT_FILE = BASE_DIR / "RAPOR.md"
CONFIG_FILE = BASE_DIR / "config.yaml"
STATIC_DIR = BASE_DIR / "static"

# Agent modülleri yükle
sys.path.insert(0, str(BASE_DIR))

app = Flask(__name__, static_folder=str(STATIC_DIR))


# ─────────────────────────────────────────────
#  Veri Katmanı
# ─────────────────────────────────────────────

def _load_tasks() -> Dict[str, Any]:
    """tasks.json dosyasını yükle."""
    if TASKS_FILE.exists():
        return json.loads(TASKS_FILE.read_text(encoding="utf-8"))
    return {"tasks": []}


def _save_tasks(data: Dict[str, Any]) -> None:
    """tasks.json dosyasına kaydet."""
    TASKS_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _find_task(task_id: str) -> tuple:
    """ID ile görev bul. (data, index) döndürür."""
    data = _load_tasks()
    for i, t in enumerate(data["tasks"]):
        if t["id"] == task_id:
            return data, i
    return data, -1


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ─────────────────────────────────────────────
#  Başlangıç Verileri (İlk çalıştırmada)
# ─────────────────────────────────────────────

INITIAL_TASKS = [
    {
        "title": "Konuşma Geçmişi Kaydetme",
        "description": "JSON tabanlı oturum kayıt/yükleme/silme sistemi eklendi.",
        "status": "completed",
        "category": "core",
        "priority": "high",
    },
    {
        "title": "Loglama Sistemi",
        "description": "RotatingFileHandler + konsol loglama altyapısı oluşturuldu.",
        "status": "completed",
        "category": "core",
        "priority": "high",
    },
    {
        "title": "Requirements.txt",
        "description": "Ana proje bağımlılıkları dosyası oluşturuldu (9 paket).",
        "status": "completed",
        "category": "core",
        "priority": "high",
    },
    {
        "title": "Unit Test Sistemi",
        "description": "pytest ile 329 test yazıldı (tüm modüller test edildi).",
        "status": "completed",
        "category": "testing",
        "priority": "high",
    },
    {
        "title": "Çoklu Model Karşılaştırma",
        "description": "En az 3 model eğitimi ve 5-fold cross validation desteği.",
        "status": "completed",
        "category": "ml",
        "priority": "medium",
    },
    {
        "title": "Görselleştirme Modülü",
        "description": "Confusion matrix, ROC curve, feature importance, korelasyon vb. grafikler.",
        "status": "completed",
        "category": "ml",
        "priority": "medium",
    },
    {
        "title": "Config.yaml Desteği",
        "description": "Merkezi yapılandırma sistemi (YAML + env + CLI katmanları).",
        "status": "completed",
        "category": "core",
        "priority": "medium",
    },
    {
        "title": "Hata Yönetimi (Exceptions)",
        "description": "7 özel hata sınıfı ile detaylı Türkçe hata mesajları.",
        "status": "completed",
        "category": "core",
        "priority": "medium",
    },
    {
        "title": "İlerleme Göstergesi (Spinner)",
        "description": "Terminal braille spinner animasyonu (context manager).",
        "status": "completed",
        "category": "ui",
        "priority": "medium",
    },
    {
        "title": "Web Arayüzü (Gradio)",
        "description": "Gradio tabanlı chat arayüzü — web_ui.py modülü.",
        "status": "completed",
        "category": "ui",
        "priority": "low",
    },
    {
        "title": "Çoklu LLM Backend",
        "description": "Ollama, OpenAI, Anthropic, Google Gemini, HuggingFace desteği.",
        "status": "completed",
        "category": "core",
        "priority": "low",
    },
    {
        "title": "Plugin Sistemi",
        "description": "Dinamik tool yükleme sistemi — plugins/ klasöründen otomatik keşif.",
        "status": "completed",
        "category": "core",
        "priority": "low",
    },
    {
        "title": "Veri Seti Kataloğu",
        "description": "15+ hazır veri seti tanımı (medikal, çevre, biyosinyal).",
        "status": "completed",
        "category": "ml",
        "priority": "low",
    },
    {
        "title": "Otomatik Rapor Oluşturucu",
        "description": "ML projelerinin otomatik Markdown raporlarını üreten modül.",
        "status": "completed",
        "category": "ml",
        "priority": "low",
    },
    {
        "title": "MLflow Entegrasyonu",
        "description": "Deney takibi wrapper — MLflow yoksa JSON fallback.",
        "status": "completed",
        "category": "ml",
        "priority": "low",
    },
    {
        "title": "Docker Desteği",
        "description": "Dockerfile ve docker-compose ile konteyner dağıtımı.",
        "status": "completed",
        "category": "devops",
        "priority": "medium",
    },
    {
        "title": "CI/CD Pipeline",
        "description": "GitHub Actions ile otomatik test ve dağıtım pipeline'ı.",
        "status": "completed",
        "category": "devops",
        "priority": "medium",
    },
    {
        "title": "Biyomühendislik Entegrasyonu",
        "description": "bioeng_toolkit modülünü agent tool'ları arasına tam entegre edildi.",
        "status": "completed",
        "category": "bioeng",
        "priority": "high",
    },
    {
        "title": "RAG Entegrasyonu",
        "description": "Retrieval-Augmented Generation ile doküman tabanlı soru-cevap.",
        "status": "completed",
        "category": "core",
        "priority": "low",
    },
    {
        "title": "Workspace Temizliği",
        "description": "workspace/workspace/ çift klasör yapısı düzeltildi.",
        "status": "completed",
        "category": "core",
        "priority": "medium",
    },
    {
        "title": "Ek Modül Testleri",
        "description": "web_ui, report_generator, plugin_manager, preprocessor testleri yazıldı.",
        "status": "completed",
        "category": "testing",
        "priority": "medium",
    },
    {
        "title": "API Modu (REST)",
        "description": "Agent'ı REST API olarak çalıştırabilme desteği.",
        "status": "completed",
        "category": "core",
        "priority": "low",
    },
    {
        "title": "Hiperparametre Optimizasyonu",
        "description": "GridSearchCV ve RandomizedSearchCV ile otomatik hiperparametre arama.",
        "status": "completed",
        "category": "ml",
        "priority": "medium",
    },
    {
        "title": "Veri Ön İşleme Pipeline",
        "description": "NaN doldurma, outlier tespiti, ölçeklendirme, PCA, polinom özellikler.",
        "status": "completed",
        "category": "ml",
        "priority": "medium",
    },
    {
        "title": "Dashboard İyileştirmeleri",
        "description": "Proje geçmişi, model karşılaştırma paneli, metrik güncellemeleri.",
        "status": "completed",
        "category": "ui",
        "priority": "medium",
    },
    {
        "title": "Model Kaydetme & Yükleme",
        "description": "Joblib ile model kaydet/yükle + standalone model_loader utility.",
        "status": "completed",
        "category": "ml",
        "priority": "high",
    },
]


def _seed_tasks() -> None:
    """İlk çalıştırmada başlangıç görevlerini oluştur."""
    if TASKS_FILE.exists():
        return  # Zaten var, dokunma

    now = _now()
    tasks = []
    for t in INITIAL_TASKS:
        task = {
            "id": uuid.uuid4().hex[:12],
            "title": t["title"],
            "description": t["description"],
            "status": t["status"],
            "category": t.get("category", "general"),
            "priority": t.get("priority", "medium"),
            "created_at": now,
            "updated_at": now,
            "approved_at": now if t["status"] == "completed" else None,
        }
        tasks.append(task)

    _save_tasks({"tasks": tasks})
    print(f"📋 {len(tasks)} başlangıç görevi oluşturuldu.")


# ─────────────────────────────────────────────
#  Modül Bilgileri
# ─────────────────────────────────────────────

def _get_modules() -> List[Dict[str, Any]]:
    """Proje modüllerinin bilgilerini topla."""
    modules = []
    py_files = [
        ("agent.py", "Ana Agent", "core"),
        ("bioeng_toolkit.py", "Biyomühendislik Araç Seti", "bioeng"),
        ("exceptions.py", "Hata Sınıfları", "core"),
        ("llm_backend.py", "Çoklu LLM Backend", "core"),
        ("plugin_manager.py", "Plugin Sistemi", "core"),
        ("dataset_catalog.py", "Veri Seti Kataloğu", "ml"),
        ("report_generator.py", "Rapor Oluşturucu", "ml"),
        ("mlflow_tracker.py", "MLflow Entegrasyonu", "ml"),
        ("web_ui.py", "Gradio Web Arayüzü", "ui"),
        ("progress.py", "Terminal Spinner", "ui"),
        ("dashboard.py", "Task Dashboard", "ui"),
        ("utils/config.py", "Yapılandırma Yönetimi", "core"),
        ("utils/model_compare.py", "Model Karşılaştırma", "ml"),
        ("utils/model_loader.py", "Model Yükleme", "ml"),
        ("utils/hyperparameter_optimizer.py", "Hiperparametre Optimizasyonu", "ml"),
        ("utils/preprocessor.py", "Veri Ön İşleme Pipeline", "ml"),
        ("utils/visualize.py", "Görselleştirme", "ml"),
    ]

    for filename, description, category in py_files:
        filepath = BASE_DIR / filename
        if filepath.exists():
            content = filepath.read_text(encoding="utf-8", errors="replace")
            lines = len(content.splitlines())
            size = filepath.stat().st_size
            modules.append({
                "filename": filename,
                "description": description,
                "category": category,
                "lines": lines,
                "size_kb": round(size / 1024, 1),
            })

    return modules


# ─────────────────────────────────────────────
#  API Endpointleri
# ─────────────────────────────────────────────

@app.route("/")
def index():
    """Dashboard HTML sayfasını sun."""
    return send_from_directory(str(STATIC_DIR), "dashboard.html")


@app.route("/favicon.ico")
def favicon():
    """Favicon isteğine yanıt ver (404 hatasını önlemek için)."""
    return send_from_directory(str(STATIC_DIR), "favicon.png", mimetype="image/png")


@app.route("/api/tasks", methods=["GET"])
def get_tasks():
    """Tüm görevleri getir. ?status=completed gibi filtre destekler."""
    data = _load_tasks()
    tasks = data["tasks"]

    # Filtreler
    status = request.args.get("status")
    category = request.args.get("category")
    priority = request.args.get("priority")
    search = request.args.get("search", "").lower()

    if status:
        tasks = [t for t in tasks if t["status"] == status]
    if category:
        tasks = [t for t in tasks if t.get("category") == category]
    if priority:
        tasks = [t for t in tasks if t.get("priority") == priority]
    if search:
        tasks = [t for t in tasks if search in t["title"].lower() or search in t.get("description", "").lower()]

    return jsonify({"tasks": tasks, "total": len(tasks)})


@app.route("/api/tasks", methods=["POST"])
def create_task():
    """Yeni görev oluştur."""
    body = request.get_json(force=True)
    if not body.get("title"):
        return jsonify({"error": "Görev başlığı gerekli."}), 400

    now = _now()
    task = {
        "id": uuid.uuid4().hex[:12],
        "title": body["title"],
        "description": body.get("description", ""),
        "status": body.get("status", "pending"),
        "category": body.get("category", "general"),
        "priority": body.get("priority", "medium"),
        "created_at": now,
        "updated_at": now,
        "approved_at": None,
    }

    data = _load_tasks()
    data["tasks"].append(task)
    _save_tasks(data)

    return jsonify(task), 201


@app.route("/api/tasks/<task_id>", methods=["PUT"])
def update_task(task_id: str):
    """Görevi güncelle."""
    data, idx = _find_task(task_id)
    if idx == -1:
        return jsonify({"error": "Görev bulunamadı."}), 404

    body = request.get_json(force=True)
    task = data["tasks"][idx]

    # İzin verilen alanları güncelle
    for field in ("title", "description", "status", "category", "priority"):
        if field in body:
            task[field] = body[field]

    task["updated_at"] = _now()

    # Eğer durum completed yapılıyorsa approved_at de güncelle
    if body.get("status") == "completed" and not task.get("approved_at"):
        task["approved_at"] = _now()

    data["tasks"][idx] = task
    _save_tasks(data)

    return jsonify(task)


@app.route("/api/tasks/<task_id>", methods=["DELETE"])
def delete_task(task_id: str):
    """Görevi sil."""
    data, idx = _find_task(task_id)
    if idx == -1:
        return jsonify({"error": "Görev bulunamadı."}), 404

    removed = data["tasks"].pop(idx)
    _save_tasks(data)

    return jsonify({"deleted": removed["id"], "title": removed["title"]})


@app.route("/api/tasks/<task_id>/approve", methods=["POST"])
def approve_task(task_id: str):
    """Görevi onayla → completed durumuna al."""
    data, idx = _find_task(task_id)
    if idx == -1:
        return jsonify({"error": "Görev bulunamadı."}), 404

    task = data["tasks"][idx]
    task["status"] = "completed"
    task["approved_at"] = _now()
    task["updated_at"] = _now()

    data["tasks"][idx] = task
    _save_tasks(data)

    return jsonify(task)


@app.route("/api/tasks/<task_id>/reject", methods=["POST"])
def reject_task(task_id: str):
    """Görevi reddet → pending durumuna geri al."""
    data, idx = _find_task(task_id)
    if idx == -1:
        return jsonify({"error": "Görev bulunamadı."}), 404

    task = data["tasks"][idx]
    task["status"] = "pending"
    task["approved_at"] = None
    task["updated_at"] = _now()

    data["tasks"][idx] = task
    _save_tasks(data)

    return jsonify(task)


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Proje istatistiklerini döndür."""
    data = _load_tasks()
    tasks = data["tasks"]

    total = len(tasks)
    completed = sum(1 for t in tasks if t["status"] == "completed")
    in_progress = sum(1 for t in tasks if t["status"] == "in_progress")
    pending = sum(1 for t in tasks if t["status"] == "pending")

    # Kategori dağılımı
    categories = {}
    for t in tasks:
        cat = t.get("category", "general")
        categories[cat] = categories.get(cat, 0) + 1

    # Modül bilgileri
    modules = _get_modules()
    total_lines = sum(m["lines"] for m in modules)

    return jsonify({
        "total": total,
        "completed": completed,
        "in_progress": in_progress,
        "pending": pending,
        "completion_pct": round(completed / total * 100, 1) if total else 0,
        "categories": categories,
        "total_lines": total_lines,
        "total_modules": len(modules),
        "total_tests": 329,
    })


@app.route("/api/report", methods=["GET"])
def get_report():
    """RAPOR.md içeriğini döndür."""
    if not REPORT_FILE.exists():
        return jsonify({"error": "Rapor dosyası bulunamadı."}), 404
    content = REPORT_FILE.read_text(encoding="utf-8")
    return jsonify({"content": content, "updated_at": _now()})


@app.route("/api/modules", methods=["GET"])
def get_modules():
    """Modül bilgilerini döndür."""
    modules = _get_modules()
    return jsonify({"modules": modules, "total_lines": sum(m["lines"] for m in modules)})


# ─────────────────────────────────────────────
#  Veri Seti Katalogu API
# ─────────────────────────────────────────────

@app.route("/api/datasets", methods=["GET"])
def api_list_datasets():
    """Veri seti kataloğunu listele."""
    try:
        from dataset_catalog import list_datasets, get_categories
        category = request.args.get("category")
        task_type = request.args.get("task_type")
        datasets = list_datasets(category=category, task_type=task_type)
        return jsonify({"datasets": datasets, "total": len(datasets), "categories": get_categories()})
    except Exception as e:
        return jsonify({"error": str(e), "datasets": []}), 500


@app.route("/api/datasets/<dataset_id>/load", methods=["POST"])
def api_load_dataset(dataset_id: str):
    """Veri setini yükle ve özet bilgilerini döndür."""
    try:
        from dataset_catalog import load_dataset
        X, y, features = load_dataset(dataset_id)
        return jsonify({
            "dataset_id": dataset_id,
            "samples": X.shape[0],
            "features": X.shape[1],
            "feature_names": list(features),
            "target_classes": len(set(y.tolist())),
            "target_distribution": {str(k): int(v) for k, v in zip(*__import__("numpy").unique(y, return_counts=True))},
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ─────────────────────────────────────────────
#  Kaydedilmiş Model API
# ─────────────────────────────────────────────

WORKSPACE_DIR = BASE_DIR / "workspace"


# ─────────────────────────────────────────────
#  Proje Geçmişi API
# ─────────────────────────────────────────────

@app.route("/api/projects", methods=["GET"])
def api_list_projects():
    """Workspace'teki ML projelerini listele."""
    projects = []
    ws = WORKSPACE_DIR
    if not ws.exists():
        return jsonify({"projects": [], "total": 0})

    for proj_dir in sorted(ws.iterdir()):
        if not proj_dir.is_dir() or proj_dir.name.startswith("."):
            continue

        # Proje bilgilerini topla
        info = {
            "id": proj_dir.name,
            "name": proj_dir.name.replace("_", " ").title(),
            "path": str(proj_dir.relative_to(BASE_DIR)),
            "created": datetime.fromtimestamp(proj_dir.stat().st_ctime).isoformat(timespec="seconds"),
            "modified": datetime.fromtimestamp(proj_dir.stat().st_mtime).isoformat(timespec="seconds"),
            "has_results": (proj_dir / "results").exists(),
            "has_report": (proj_dir / "report.md").exists() or (proj_dir / "README.md").exists(),
            "has_model": any(proj_dir.rglob("*.pkl")),
            "file_count": sum(1 for _ in proj_dir.rglob("*") if _.is_file()),
        }

        # Sonuç dosyası varsa metrikleri oku
        results_json = proj_dir / "results" / "comparison_results.json"
        if results_json.exists():
            try:
                rdata = json.loads(results_json.read_text(encoding="utf-8"))
                # JSON array mı dict mi kontrol et
                if isinstance(rdata, list):
                    info["model_count"] = len(rdata)
                    # En iyi modeli bul (ilk metrik değerine göre)
                    if rdata:
                        name_key = "Model" if "Model" in rdata[0] else "model_name"
                        score_keys = [k for k in rdata[0] if isinstance(rdata[0][k], (int, float))]
                        if score_keys:
                            best = max(rdata, key=lambda r: r.get(score_keys[0], 0))
                            info["best_model"] = best.get(name_key, "")
                else:
                    info["best_model"] = rdata.get("best_model", "")
                    info["model_count"] = len(rdata.get("results", []))
            except Exception:
                pass

        projects.append(info)

    projects.sort(key=lambda p: p["modified"], reverse=True)
    return jsonify({"projects": projects, "total": len(projects)})


@app.route("/api/projects/<project_id>/results", methods=["GET"])
def api_project_results(project_id: str):
    """Belirli bir projenin sonuçlarını getir."""
    proj_dir = WORKSPACE_DIR / project_id
    if not proj_dir.exists():
        return jsonify({"error": "Proje bulunamadı."}), 404

    result = {
        "project_id": project_id,
        "name": project_id.replace("_", " ").title(),
        "files": [],
        "models": [],
        "report": None,
        "plots": [],
        "comparison": None,
    }

    # Dosya listesi (ilk 50)
    for f in sorted(proj_dir.rglob("*")):
        if f.is_file() and not f.name.startswith("."):
            result["files"].append({
                "path": str(f.relative_to(proj_dir)),
                "size_kb": round(f.stat().st_size / 1024, 1),
            })
            if len(result["files"]) >= 50:
                break

    # Model dosyaları
    for pkl in proj_dir.rglob("*.pkl"):
        meta_path = pkl.with_name(pkl.stem + "_meta.json")
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        result["models"].append({
            "name": meta.get("model_name", pkl.stem),
            "path": str(pkl.relative_to(proj_dir)),
            "metrics": meta.get("metrics", {}),
        })

    # Rapor
    for report_name in ("report.md", "README.md"):
        rp = proj_dir / report_name
        if rp.exists():
            result["report"] = rp.read_text(encoding="utf-8", errors="replace")[:10000]
            break

    # Grafikler
    plots_dir = proj_dir / "results" / "plots"
    if plots_dir.exists():
        for img in sorted(plots_dir.glob("*.png")):
            result["plots"].append({
                "name": img.stem.replace("_", " ").title(),
                "path": str(img.relative_to(proj_dir)),
            })

    # Karşılaştırma sonuçları
    cmp = proj_dir / "results" / "comparison_results.json"
    if cmp.exists():
        try:
            cmp_data = json.loads(cmp.read_text(encoding="utf-8"))
            # Normalize: array ise dict'e çevir
            if isinstance(cmp_data, list):
                result["comparison"] = {"results": cmp_data}
            else:
                result["comparison"] = cmp_data
        except Exception:
            pass

    return jsonify(result)


@app.route("/api/compare", methods=["GET"])
def api_compare_models():
    """Tüm projelerdeki model karşılaştırma verilerini topluca getir."""
    comparisons = []
    ws = WORKSPACE_DIR
    if not ws.exists():
        return jsonify({"comparisons": [], "total": 0})

    for proj_dir in sorted(ws.iterdir()):
        if not proj_dir.is_dir():
            continue
        cmp_file = proj_dir / "results" / "comparison_results.json"
        if not cmp_file.exists():
            continue
        try:
            data = json.loads(cmp_file.read_text(encoding="utf-8"))
            # Array ve dict formatını destekle
            if isinstance(data, list):
                results_list = data
                name_key = "Model" if data and "Model" in data[0] else "model_name"
                score_keys = [k for k in (data[0] if data else {}) if isinstance((data[0] if data else {}).get(k), (int, float))]
                best = ""
                if data and score_keys:
                    best_item = max(data, key=lambda r: r.get(score_keys[0], 0))
                    best = best_item.get(name_key, "")
                comparisons.append({
                    "project": proj_dir.name,
                    "best_model": best,
                    "task_type": "classification",
                    "results": results_list,
                    "metric_names": list(data[0].keys()) if data else [],
                })
            else:
                comparisons.append({
                    "project": proj_dir.name,
                    "best_model": data.get("best_model", ""),
                    "task_type": data.get("task_type", "unknown"),
                    "results": data.get("results", []),
                    "metric_names": list(data.get("results", [{}])[0].keys()) if data.get("results") else [],
                })
        except Exception:
            continue

    return jsonify({"comparisons": comparisons, "total": len(comparisons)})

@app.route("/api/models", methods=["GET"])
def api_list_models():
    """Workspace altındaki tüm .pkl model dosyalarını listele."""
    models = []
    search_dirs = [WORKSPACE_DIR, BASE_DIR / "results"]
    for sdir in search_dirs:
        if sdir.exists():
            for pkl in sdir.rglob("*.pkl"):
                meta_path = pkl.with_name(pkl.stem + "_meta.json")
                meta = {}
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    except Exception:
                        pass
                models.append({
                    "path": str(pkl.relative_to(BASE_DIR)),
                    "name": meta.get("model_name", pkl.stem),
                    "task_type": meta.get("task_type", "unknown"),
                    "metrics": meta.get("metrics", {}),
                    "size_kb": round(pkl.stat().st_size / 1024, 1),
                    "created": datetime.fromtimestamp(pkl.stat().st_mtime).isoformat(timespec="seconds"),
                })
    models.sort(key=lambda m: m["created"], reverse=True)
    return jsonify({"models": models, "total": len(models)})


@app.route("/api/models/info", methods=["POST"])
def api_model_info():
    """Model meta bilgilerini getir."""
    body = request.get_json(force=True)
    model_path = body.get("path", "")
    full_path = str(BASE_DIR / model_path)
    try:
        from utils.model_loader import model_info
        info = model_info(full_path)
        return jsonify(info)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


def _load_config() -> dict:
    """config.yaml dosyasını yükle."""
    if CONFIG_FILE.exists():
        return yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8")) or {}
    return {}


def _save_config(cfg: dict) -> None:
    """config.yaml dosyasına kaydet."""
    CONFIG_FILE.write_text(
        yaml.dump(cfg, default_flow_style=False, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


@app.route("/api/config", methods=["GET"])
def get_config():
    """Mevcut yapılandırmayı döndür."""
    cfg = _load_config()
    return jsonify(cfg)


@app.route("/api/config", methods=["PUT"])
def update_config():
    """Yapılandırmayı güncelle."""
    body = request.get_json(force=True)
    cfg = _load_config()

    # Gelen key/value çiftlerini güncelle (nested)
    def deep_update(base, updates):
        for k, v in updates.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                deep_update(base[k], v)
            else:
                base[k] = v

    deep_update(cfg, body)
    _save_config(cfg)
    return jsonify({"status": "ok", "config": cfg})


@app.route("/api/config/api-keys", methods=["GET"])
def get_api_keys():
    """Mevcut API key durumlarını döndür (değerleri gizli)."""
    keys = {
        "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
        "ANTHROPIC_API_KEY": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "GOOGLE_API_KEY": bool(os.environ.get("GOOGLE_API_KEY")),
        "GEMINI_API_KEY": bool(os.environ.get("GEMINI_API_KEY")),
        "HF_API_TOKEN": bool(os.environ.get("HF_API_TOKEN")),
    }
    return jsonify(keys)


@app.route("/api/config/api-keys", methods=["PUT"])
def set_api_keys():
    """API key'leri ortam değişkenlerine kaydet."""
    body = request.get_json(force=True)
    allowed = {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "HF_API_TOKEN"}
    updated = []
    for key, value in body.items():
        if key in allowed and value:
            os.environ[key] = value
            updated.append(key)
    return jsonify({"status": "ok", "updated": updated})


@app.route("/api/ollama/models", methods=["GET"])
def get_ollama_models():
    """Ollama'da yüklü modelleri listele."""
    try:
        import ollama as ollama_lib
        models = ollama_lib.list()
        model_list = []
        for m in models.get("models", []):
            model_list.append({
                "name": m.get("name", ""),
                "size_gb": round(m.get("size", 0) / (1024**3), 1),
                "modified": m.get("modified_at", ""),
            })
        return jsonify({"models": model_list})
    except ImportError:
        return jsonify({"models": [], "error": "ollama kütüphanesi yüklü değil."})
    except Exception as e:
        return jsonify({"models": [], "error": str(e)})


# ─────────────────────────────────────────────
#  Agent Chat API
# ─────────────────────────────────────────────

_agent_lock = threading.Lock()


@app.route("/api/agent/chat", methods=["POST"])
def agent_chat():
    """Agent'a mesaj gönder ve yanıt al."""
    body = request.get_json(force=True)
    user_msg = body.get("message", "").strip()
    if not user_msg:
        return jsonify({"error": "Mesaj boş olamaz."}), 400

    model = body.get("model", "")
    cfg = _load_config()
    if not model:
        model = cfg.get("agent", {}).get("model", "qwen2.5:7b-instruct")

    timeout = cfg.get("agent", {}).get("timeout", 180)
    max_steps = body.get("max_steps", cfg.get("agent", {}).get("max_steps", 10))

    # Agent'ı çağır
    try:
        from agent import (
            SYSTEM_PROMPT, extract_tool, run_python, run_bash,
            web_search, web_open, read_file, write_file, append_todo,
            AgentConfig,
        )
        from llm_backend import auto_create_backend
    except ImportError as e:
        return jsonify({"error": f"Agent modülleri yüklenemedi: {e}"}), 500

    workspace = Path(cfg.get("workspace", {}).get("base_dir", "workspace")).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    steps = []
    final_response = ""

    with _agent_lock:
        for step_i in range(max_steps):
            try:
                backend_mode = cfg.get("agent", {}).get("backend", "auto")
                backend = auto_create_backend(model, mode=backend_mode)
                assistant = backend.chat(messages)
            except Exception as e:
                final_response = f"❌ LLM Hatası: {e}"
                steps.append({"type": "error", "content": str(e)})
                break

            tool, payload, outside = extract_tool(assistant)

            # Tool yoksa düz yanıt
            if tool is None:
                final_response = assistant
                steps.append({"type": "response", "content": assistant})
                break

            # Tool varsa çalıştır
            if outside:
                steps.append({"type": "text", "content": outside})

            try:
                agent_cfg = AgentConfig(
                    model=model, workspace=workspace,
                    timeout=timeout, max_steps=max_steps,
                    history_dir=Path("conversation_history"),
                )
                tool_map = {
                    "PYTHON": lambda p: run_python(p, workspace, timeout_s=timeout),
                    "BASH": lambda p: run_bash(p, workspace, timeout_s=timeout),
                    "WEB_SEARCH": lambda p: web_search(p),
                    "WEB_OPEN": lambda p: web_open(p),
                    "READ_FILE": lambda p: read_file(p, workspace),
                    "WRITE_FILE": lambda p: write_file(p, workspace),
                    "TODO": lambda p: append_todo(p, workspace),
                }
                runner = tool_map.get(tool)
                if runner:
                    out = runner(payload)
                else:
                    out = f"Bilinmeyen tool: {tool}"
            except Exception as e:
                out = f"Tool hatası: {e}"

            steps.append({"type": "tool", "tool": tool, "output": out[:2000]})

            messages.append({"role": "assistant", "content": assistant})
            messages.append({
                "role": "user",
                "content": f"TOOL_OUTPUT ({tool}):\n{out}\n\nContinue. If done, answer normally (no tool).",
            })

            final_response = outside or ""

    return jsonify({
        "response": final_response,
        "steps": steps,
        "model": model,
        "total_steps": len(steps),
    })


# ─────────────────────────────────────────────
#  Audit Log API (Pillar 5: Governance)
# ─────────────────────────────────────────────

AUDIT_LOG_FILE = BASE_DIR / "logs" / "agent.log"


@app.route("/api/audit", methods=["GET"])
def api_audit_log():
    """Ajan denetim izlerini (audit trail) döndürür.

    Parametreler:
        ?limit=50       — Son N satır (varsayılan 50)
        ?filter=HITL    — Sadece belirli türdeki logları filtrele
                         (HITL, PYTHON, BASH, BROWSER, APPROVAL, ERROR)
    """
    limit = min(int(request.args.get("limit", 50)), 500)
    log_filter = request.args.get("filter", "").upper()

    entries = []
    if not AUDIT_LOG_FILE.exists():
        return jsonify({"entries": [], "total": 0, "note": "Log dosyası bulunamadı."})

    try:
        lines = AUDIT_LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
        # Son N satırı al (en yeni önce)
        recent = lines[-limit * 3:]  # Filtre için fazla satır al
        recent.reverse()

        filter_keywords = {
            "HITL": ["HITL", "approval", "onay", "APPROVAL"],
            "PYTHON": ["PYTHON", "run_python", "sandbox"],
            "BASH": ["BASH", "run_bash", "subprocess"],
            "BROWSER": ["BROWSER", "browser_agent", "playwright"],
            "ERROR": ["ERROR", "Exception", "Traceback", "hata"],
            "APPROVAL": ["HITL", "approval", "approved", "rejected"],
        }

        keywords = filter_keywords.get(log_filter, []) if log_filter else []

        for line in recent:
            if len(entries) >= limit:
                break
            line = line.strip()
            if not line:
                continue

            # Filtre uygulanacaksa kontrol et
            if keywords and not any(kw.lower() in line.lower() for kw in keywords):
                continue

            # Log satırını ayrıştır
            entry = {"raw": line[:500]}  # Çok uzun logları kes

            # Zaman damgası ve seviye çıkar
            # Örnek format: "2026-03-04 19:11:51 [WARNING] mesaj"
            parts = line.split(" ", 3)
            if len(parts) >= 3:
                entry["timestamp"] = f"{parts[0]} {parts[1]}" if parts[0][0:2] == "20" else ""
                # Seviye çıkar [INFO], [WARNING], [ERROR]
                for p in parts:
                    if p.startswith("[") and p.endswith("]"):
                        entry["level"] = p.strip("[]")
                        break

            entries.append(entry)

    except Exception as e:
        return jsonify({"entries": [], "total": 0, "error": str(e)}), 500

    return jsonify({
        "entries": entries,
        "total": len(entries),
        "log_file": str(AUDIT_LOG_FILE),
        "available_filters": ["HITL", "PYTHON", "BASH", "BROWSER", "ERROR", "APPROVAL"],
    })


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    _seed_tasks()
    print(f"🚀 Dashboard çalışıyor: http://localhost:5050")
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() in ("true", "1", "yes")
    app.run(host="127.0.0.1", port=5050, debug=debug_mode)
