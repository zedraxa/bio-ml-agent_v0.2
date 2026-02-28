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
        "description": "pytest ile 159 test yazıldı (agent, exceptions, progress).",
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
        "status": "pending",
        "category": "devops",
        "priority": "medium",
    },
    {
        "title": "CI/CD Pipeline",
        "description": "GitHub Actions ile otomatik test ve dağıtım pipeline'ı.",
        "status": "pending",
        "category": "devops",
        "priority": "medium",
    },
    {
        "title": "Biyomühendislik Entegrasyonu",
        "description": "bioeng_toolkit modülünü agent tool'ları arasına tam entegre et.",
        "status": "in_progress",
        "category": "bioeng",
        "priority": "high",
    },
    {
        "title": "RAG Entegrasyonu",
        "description": "Retrieval-Augmented Generation ile doküman tabanlı soru-cevap.",
        "status": "pending",
        "category": "core",
        "priority": "low",
    },
    {
        "title": "Workspace Temizliği",
        "description": "workspace/workspace/ çift klasör yapısını düzelt ve organize et.",
        "status": "pending",
        "category": "core",
        "priority": "medium",
    },
    {
        "title": "Ek Modül Testleri",
        "description": "web_ui, mlflow_tracker, report_generator için unit testler yaz.",
        "status": "pending",
        "category": "testing",
        "priority": "medium",
    },
    {
        "title": "API Modu (REST)",
        "description": "Agent'ı REST API olarak çalıştırabilme desteği.",
        "status": "pending",
        "category": "core",
        "priority": "low",
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
        "total_tests": 257,
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
#  Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    _seed_tasks()

    print("╔══════════════════════════════════════════════════╗")
    print("║   🧠 Bio-ML Agent — Task Dashboard              ║")
    print("║   📍 http://localhost:5050                       ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    app.run(host="0.0.0.0", port=5050, debug=True)
