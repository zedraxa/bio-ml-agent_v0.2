# services/dashboard_service.py
# ═══════════════════════════════════════════════════════════
#  Dashboard iş mantığı — dashboard.py'den taşındı.
#  web_ui.py Gradio tabları ve api_server.py bu servisi kullanır.
# ═══════════════════════════════════════════════════════════

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# ─────────────────────────────────────────────
#  Yollar
# ─────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
TASKS_FILE = BASE_DIR / "tasks.json"
REPORT_FILE = BASE_DIR / "RAPOR.md"
CONFIG_FILE = BASE_DIR / "config.yaml"
WORKSPACE_DIR = BASE_DIR / "workspace"
AUDIT_LOG_FILE = BASE_DIR / "logs" / "agent.log"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ═══════════════════════════════════════════════════════════
#  Görev Yönetimi (Task CRUD)
# ═══════════════════════════════════════════════════════════

INITIAL_TASKS = [
    {"title": "Konuşma Geçmişi Kaydetme", "description": "JSON tabanlı oturum kayıt/yükleme/silme sistemi eklendi.", "status": "completed", "category": "core", "priority": "high"},
    {"title": "Loglama Sistemi", "description": "RotatingFileHandler + konsol loglama altyapısı oluşturuldu.", "status": "completed", "category": "core", "priority": "high"},
    {"title": "Requirements.txt", "description": "Ana proje bağımlılıkları dosyası oluşturuldu (9 paket).", "status": "completed", "category": "core", "priority": "high"},
    {"title": "Unit Test Sistemi", "description": "pytest ile 329 test yazıldı (tüm modüller test edildi).", "status": "completed", "category": "testing", "priority": "high"},
    {"title": "Çoklu Model Karşılaştırma", "description": "En az 3 model eğitimi ve 5-fold cross validation desteği.", "status": "completed", "category": "ml", "priority": "medium"},
    {"title": "Görselleştirme Modülü", "description": "Confusion matrix, ROC curve, feature importance grafikleri.", "status": "completed", "category": "ml", "priority": "medium"},
    {"title": "Config.yaml Desteği", "description": "Merkezi yapılandırma sistemi (YAML + env + CLI).", "status": "completed", "category": "core", "priority": "medium"},
    {"title": "Hata Yönetimi (Exceptions)", "description": "7 özel hata sınıfı ile detaylı Türkçe hata mesajları.", "status": "completed", "category": "core", "priority": "medium"},
    {"title": "İlerleme Göstergesi (Spinner)", "description": "Terminal braille spinner animasyonu.", "status": "completed", "category": "ui", "priority": "medium"},
    {"title": "Web Arayüzü (Gradio)", "description": "Gradio tabanlı chat arayüzü.", "status": "completed", "category": "ui", "priority": "low"},
    {"title": "Çoklu LLM Backend", "description": "Ollama, OpenAI, Anthropic, Google Gemini desteği.", "status": "completed", "category": "core", "priority": "low"},
    {"title": "Plugin Sistemi", "description": "Dinamik tool yükleme sistemi.", "status": "completed", "category": "core", "priority": "low"},
    {"title": "Veri Seti Kataloğu", "description": "15+ hazır veri seti tanımı.", "status": "completed", "category": "ml", "priority": "low"},
    {"title": "Otomatik Rapor Oluşturucu", "description": "ML projelerinin otomatik Markdown raporları.", "status": "completed", "category": "ml", "priority": "low"},
    {"title": "MLflow Entegrasyonu", "description": "Deney takibi wrapper.", "status": "completed", "category": "ml", "priority": "low"},
    {"title": "Docker Desteği", "description": "Dockerfile ve docker-compose ile konteyner dağıtımı.", "status": "completed", "category": "devops", "priority": "medium"},
    {"title": "CI/CD Pipeline", "description": "GitHub Actions ile otomatik test ve dağıtım.", "status": "completed", "category": "devops", "priority": "medium"},
    {"title": "Biyomühendislik Entegrasyonu", "description": "bioeng_toolkit agent tool'ları arasında.", "status": "completed", "category": "bioeng", "priority": "high"},
    {"title": "RAG Entegrasyonu", "description": "Doküman tabanlı soru-cevap.", "status": "completed", "category": "core", "priority": "low"},
    {"title": "Model Kaydetme & Yükleme", "description": "Joblib ile model kaydet/yükle.", "status": "completed", "category": "ml", "priority": "high"},
]


def seed_tasks() -> None:
    """İlk çalıştırmada başlangıç görevlerini oluştur."""
    if TASKS_FILE.exists():
        return
    now = _now()
    tasks = []
    for t in INITIAL_TASKS:
        tasks.append({
            "id": uuid.uuid4().hex[:12],
            "title": t["title"],
            "description": t["description"],
            "status": t["status"],
            "category": t.get("category", "general"),
            "priority": t.get("priority", "medium"),
            "created_at": now,
            "updated_at": now,
            "approved_at": now if t["status"] == "completed" else None,
        })
    TASKS_FILE.write_text(json.dumps({"tasks": tasks}, ensure_ascii=False, indent=2), encoding="utf-8")


def load_tasks() -> Dict[str, Any]:
    """Görev listesini yükle."""
    if TASKS_FILE.exists():
        return json.loads(TASKS_FILE.read_text(encoding="utf-8"))
    return {"tasks": []}


def save_tasks(data: Dict[str, Any]) -> None:
    """Görev listesini kaydet."""
    TASKS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def find_task(task_id: str) -> Tuple[Dict[str, Any], int]:
    """ID ile görev bul. (data, index) döndürür; bulunamazsa index=-1."""
    data = load_tasks()
    for i, t in enumerate(data["tasks"]):
        if t["id"] == task_id:
            return data, i
    return data, -1


def list_tasks(status: str = None, category: str = None, priority: str = None, search: str = "") -> List[Dict]:
    """Filtrelenmiş görev listesi döndür."""
    data = load_tasks()
    tasks = data["tasks"]
    if status:
        tasks = [t for t in tasks if t["status"] == status]
    if category:
        tasks = [t for t in tasks if t.get("category") == category]
    if priority:
        tasks = [t for t in tasks if t.get("priority") == priority]
    if search:
        s = search.lower()
        tasks = [t for t in tasks if s in t["title"].lower() or s in t.get("description", "").lower()]
    return tasks


def create_task(title: str, description: str = "", status: str = "pending",
                category: str = "general", priority: str = "medium") -> Dict:
    """Yeni görev oluştur."""
    now = _now()
    task = {
        "id": uuid.uuid4().hex[:12],
        "title": title,
        "description": description,
        "status": status,
        "category": category,
        "priority": priority,
        "created_at": now,
        "updated_at": now,
        "approved_at": None,
    }
    data = load_tasks()
    data["tasks"].append(task)
    save_tasks(data)
    return task


def update_task(task_id: str, **fields) -> Optional[Dict]:
    """Görevi güncelle. Görev bulunamazsa None döndürür."""
    data, idx = find_task(task_id)
    if idx == -1:
        return None
    task = data["tasks"][idx]
    for field in ("title", "description", "status", "category", "priority"):
        if field in fields:
            task[field] = fields[field]
    task["updated_at"] = _now()
    if fields.get("status") == "completed" and not task.get("approved_at"):
        task["approved_at"] = _now()
    data["tasks"][idx] = task
    save_tasks(data)
    return task


def delete_task(task_id: str) -> Optional[Dict]:
    """Görevi sil. Bulunamazsa None."""
    data, idx = find_task(task_id)
    if idx == -1:
        return None
    removed = data["tasks"].pop(idx)
    save_tasks(data)
    return removed


def approve_task(task_id: str) -> Optional[Dict]:
    """Görevi onayla → completed."""
    return update_task(task_id, status="completed")


def reject_task(task_id: str) -> Optional[Dict]:
    """Görevi reddet → pending."""
    data, idx = find_task(task_id)
    if idx == -1:
        return None
    task = data["tasks"][idx]
    task["status"] = "pending"
    task["approved_at"] = None
    task["updated_at"] = _now()
    data["tasks"][idx] = task
    save_tasks(data)
    return task


# ═══════════════════════════════════════════════════════════
#  Proje & Model Bilgileri
# ═══════════════════════════════════════════════════════════

def list_projects() -> List[Dict[str, Any]]:
    """Workspace'teki ML projelerini listele."""
    projects = []
    if not WORKSPACE_DIR.exists():
        return projects
    for proj_dir in sorted(WORKSPACE_DIR.iterdir()):
        if not proj_dir.is_dir() or proj_dir.name.startswith("."):
            continue
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
        # Sonuç dosyası
        results_json = proj_dir / "results" / "comparison_results.json"
        if results_json.exists():
            try:
                rdata = json.loads(results_json.read_text(encoding="utf-8"))
                if isinstance(rdata, list):
                    info["model_count"] = len(rdata)
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
    return projects


def get_project_results(project_id: str) -> Optional[Dict[str, Any]]:
    """Belirli bir projenin detaylı sonuçlarını getir."""
    proj_dir = WORKSPACE_DIR / project_id
    if not proj_dir.exists():
        return None
    result = {
        "project_id": project_id,
        "name": project_id.replace("_", " ").title(),
        "files": [], "models": [], "report": None, "plots": [], "comparison": None,
    }
    # Dosyalar
    for f in sorted(proj_dir.rglob("*")):
        if f.is_file() and not f.name.startswith("."):
            result["files"].append({"path": str(f.relative_to(proj_dir)), "size_kb": round(f.stat().st_size / 1024, 1)})
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
        result["models"].append({"name": meta.get("model_name", pkl.stem), "path": str(pkl.relative_to(proj_dir)), "metrics": meta.get("metrics", {})})
    # Rapor
    for rname in ("report.md", "README.md"):
        rp = proj_dir / rname
        if rp.exists():
            result["report"] = rp.read_text(encoding="utf-8", errors="replace")[:10000]
            break
    # Grafikler
    plots_dir = proj_dir / "results" / "plots"
    if plots_dir.exists():
        for img in sorted(plots_dir.glob("*.png")):
            result["plots"].append({"name": img.stem.replace("_", " ").title(), "path": str(img.relative_to(proj_dir))})
    # Karşılaştırma
    cmp = proj_dir / "results" / "comparison_results.json"
    if cmp.exists():
        try:
            cmp_data = json.loads(cmp.read_text(encoding="utf-8"))
            result["comparison"] = {"results": cmp_data} if isinstance(cmp_data, list) else cmp_data
        except Exception:
            pass
    return result


def compare_all_models() -> List[Dict[str, Any]]:
    """Tüm projelerdeki model karşılaştırma verilerini topla."""
    comparisons = []
    if not WORKSPACE_DIR.exists():
        return comparisons
    for proj_dir in sorted(WORKSPACE_DIR.iterdir()):
        if not proj_dir.is_dir():
            continue
        cmp_file = proj_dir / "results" / "comparison_results.json"
        if not cmp_file.exists():
            continue
        try:
            data = json.loads(cmp_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                name_key = "Model" if data and "Model" in data[0] else "model_name"
                score_keys = [k for k in (data[0] if data else {}) if isinstance((data[0] if data else {}).get(k), (int, float))]
                best = ""
                if data and score_keys:
                    best = max(data, key=lambda r: r.get(score_keys[0], 0)).get(name_key, "")
                comparisons.append({"project": proj_dir.name, "best_model": best, "results": data})
            else:
                comparisons.append({"project": proj_dir.name, "best_model": data.get("best_model", ""), "results": data.get("results", [])})
        except Exception:
            continue
    return comparisons


def list_models() -> List[Dict[str, Any]]:
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
    return models


# ═══════════════════════════════════════════════════════════
#  İstatistikler & Modül Bilgileri
# ═══════════════════════════════════════════════════════════

MODULE_REGISTRY = [
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
    ("utils/config.py", "Yapılandırma Yönetimi", "core"),
    ("utils/model_compare.py", "Model Karşılaştırma", "ml"),
    ("utils/model_loader.py", "Model Yükleme", "ml"),
    ("utils/hyperparameter_optimizer.py", "Hiperparametre Optimizasyonu", "ml"),
    ("utils/preprocessor.py", "Veri Ön İşleme Pipeline", "ml"),
    ("utils/visualize.py", "Görselleştirme", "ml"),
]


def get_modules() -> List[Dict[str, Any]]:
    """Proje modüllerinin bilgilerini topla."""
    modules = []
    for filename, description, category in MODULE_REGISTRY:
        filepath = BASE_DIR / filename
        if filepath.exists():
            content = filepath.read_text(encoding="utf-8", errors="replace")
            modules.append({
                "filename": filename,
                "description": description,
                "category": category,
                "lines": len(content.splitlines()),
                "size_kb": round(filepath.stat().st_size / 1024, 1),
            })
    return modules


def get_stats() -> Dict[str, Any]:
    """Proje istatistiklerini döndür."""
    data = load_tasks()
    tasks = data["tasks"]
    total = len(tasks)
    completed = sum(1 for t in tasks if t["status"] == "completed")
    in_progress = sum(1 for t in tasks if t["status"] == "in_progress")
    pending = sum(1 for t in tasks if t["status"] == "pending")
    categories = {}
    for t in tasks:
        cat = t.get("category", "general")
        categories[cat] = categories.get(cat, 0) + 1
    modules = get_modules()
    return {
        "total": total,
        "completed": completed,
        "in_progress": in_progress,
        "pending": pending,
        "completion_pct": round(completed / total * 100, 1) if total else 0,
        "categories": categories,
        "total_lines": sum(m["lines"] for m in modules),
        "total_modules": len(modules),
    }


# ═══════════════════════════════════════════════════════════
#  Rapor & Yapılandırma
# ═══════════════════════════════════════════════════════════

def get_report() -> Optional[str]:
    """RAPOR.md içeriğini döndür."""
    if REPORT_FILE.exists():
        return REPORT_FILE.read_text(encoding="utf-8")
    return None


def load_config() -> Dict:
    """config.yaml dosyasını yükle."""
    if CONFIG_FILE.exists():
        return yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8")) or {}
    return {}


def save_config(cfg: Dict) -> None:
    """config.yaml dosyasına kaydet."""
    CONFIG_FILE.write_text(
        yaml.dump(cfg, default_flow_style=False, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def update_config(updates: Dict) -> Dict:
    """Yapılandırmayı güncelle (nested deep update)."""
    cfg = load_config()

    def deep_update(base, upd):
        for k, v in upd.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                deep_update(base[k], v)
            else:
                base[k] = v

    deep_update(cfg, updates)
    save_config(cfg)
    return cfg


def get_api_keys_status() -> Dict[str, bool]:
    """API key durumlarını döndür (değerler gizli)."""
    return {
        "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
        "ANTHROPIC_API_KEY": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "GOOGLE_API_KEY": bool(os.environ.get("GOOGLE_API_KEY")),
        "GEMINI_API_KEY": bool(os.environ.get("GEMINI_API_KEY")),
        "HF_API_TOKEN": bool(os.environ.get("HF_API_TOKEN")),
    }


# ═══════════════════════════════════════════════════════════
#  Denetim ve Telemetri
# ═══════════════════════════════════════════════════════════

def get_audit_log(limit: int = 50, log_filter: str = "") -> Dict[str, Any]:
    """Ajan denetim izlerini döndür."""
    limit = min(limit, 500)
    entries: List[Dict] = []
    if not AUDIT_LOG_FILE.exists():
        return {"entries": [], "total": 0, "note": "Log dosyası bulunamadı."}

    filter_keywords = {
        "HITL": ["HITL", "approval", "onay", "APPROVAL"],
        "PYTHON": ["PYTHON", "run_python", "sandbox"],
        "BASH": ["BASH", "run_bash", "subprocess"],
        "BROWSER": ["BROWSER", "browser_agent", "playwright"],
        "ERROR": ["ERROR", "Exception", "Traceback", "hata"],
        "APPROVAL": ["HITL", "approval", "approved", "rejected"],
    }

    try:
        lines = AUDIT_LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
        recent = lines[-(limit * 3):]
        recent.reverse()
        keywords = filter_keywords.get(log_filter.upper(), []) if log_filter else []

        for line in recent:
            if len(entries) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            if keywords and not any(kw.lower() in line.lower() for kw in keywords):
                continue
            entry: Dict[str, Any] = {"raw": line[:500]}
            parts = line.split(" ", 3)
            if len(parts) >= 3:
                entry["timestamp"] = f"{parts[0]} {parts[1]}" if parts[0][:2] == "20" else ""
                for p in parts:
                    if p.startswith("[") and p.endswith("]"):
                        entry["level"] = p.strip("[]")
                        break
            entries.append(entry)
    except Exception as e:
        return {"entries": [], "total": 0, "error": str(e)}

    return {
        "entries": entries,
        "total": len(entries),
        "available_filters": ["HITL", "PYTHON", "BASH", "BROWSER", "ERROR", "APPROVAL"],
    }
