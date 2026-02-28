# api_server.py
# ═══════════════════════════════════════════════════════════
#  Bio-ML Agent — REST API Sunucusu
#
#  Kullanım:
#    python api_server.py                          # varsayılan (port 8000)
#    python api_server.py --port 9000 --model gemini-2.5-flash
#
#  Endpoint'ler:
#    POST /api/chat          — Agent ile sohbet
#    GET  /api/datasets      — Veri seti kataloğu
#    POST /api/datasets/load — Veri seti yükleme
#    GET  /api/models        — Kaydedilmiş modeller
#    GET  /api/backends      — Kullanılabilir LLM backend'ler
#    GET  /api/config        — Yapılandırma
#    PUT  /api/config        — Yapılandırmayı güncelle
#    GET  /api/health        — Sunucu durumu
# ═══════════════════════════════════════════════════════════

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from flask import Flask, jsonify, request
from flask_cors import CORS

# ─────────────────────────────────────────────
#  Konfigürasyon
# ─────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

app = Flask(__name__)
CORS(app)

# Sunucu durumu
_server_start = datetime.now()
_request_count = 0
_lock = threading.Lock()


# ─────────────────────────────────────────────
#  Yardımcılar
# ─────────────────────────────────────────────

def _load_config() -> dict:
    config_path = BASE_DIR / "config.yaml"
    if config_path.exists():
        return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return {}


def _save_config(cfg: dict) -> None:
    config_path = BASE_DIR / "config.yaml"
    config_path.write_text(
        yaml.dump(cfg, default_flow_style=False, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _json_error(message: str, status: int = 400) -> tuple:
    return jsonify({"error": message, "status": status}), status


def _increment_requests():
    global _request_count
    with _lock:
        _request_count += 1


# ─────────────────────────────────────────────
#  Hata Yönetimi
# ─────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return _json_error("Endpoint bulunamadı.", 404)


@app.errorhandler(405)
def method_not_allowed(e):
    return _json_error("HTTP metodu desteklenmiyor.", 405)


@app.errorhandler(500)
def internal_error(e):
    return _json_error(f"Sunucu hatası: {e}", 500)


# ═══════════════════════════════════════════════════════════
#  API Endpoint'leri
# ═══════════════════════════════════════════════════════════


# ─────────────────────────────────────────────
#  Health Check
# ─────────────────────────────────────────────

@app.route("/favicon.ico")
def favicon():
    return jsonify({"status": "ok"}), 200

@app.route("/api/health", methods=["GET"])
def health():
    """Sunucu sağlık durumu."""
    _increment_requests()
    uptime = (datetime.now() - _server_start).total_seconds()
    return jsonify({
        "status": "ok",
        "version": "2.0",
        "uptime_seconds": round(uptime),
        "total_requests": _request_count,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })


# ─────────────────────────────────────────────
#  Chat (Agent İle Sohbet)
# ─────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Agent'a mesaj gönder.

    Body JSON:
        {
            "message": "Breast cancer veri setiyle bir analiz yap",
            "model": "gemini-2.5-flash",     // opsiyonel
            "max_steps": 10,                 // opsiyonel
            "session_id": "abc123"           // opsiyonel
        }

    Response JSON:
        {
            "response": "...",
            "steps": [...],
            "model": "...",
            "session_id": "...",
            "total_steps": 5
        }
    """
    _increment_requests()
    body = request.get_json(force=True)
    user_msg = body.get("message", "").strip()

    if not user_msg:
        return _json_error("'message' alanı boş olamaz.")

    cfg = _load_config()
    model = body.get("model", "") or cfg.get("agent", {}).get("model", "qwen2.5:7b-instruct")
    timeout = cfg.get("agent", {}).get("timeout", 180)
    max_steps = body.get("max_steps", cfg.get("agent", {}).get("max_steps", 10))
    session_id = body.get("session_id", uuid.uuid4().hex[:12])

    try:
        from agent import (
            SYSTEM_PROMPT, extract_tool, run_python, run_bash,
            web_search, web_open, read_file, write_file, append_todo,
            AgentConfig,
        )
        from llm_backend import auto_create_backend
    except ImportError as e:
        return _json_error(f"Agent modülleri yüklenemedi: {e}", 500)

    workspace = Path(cfg.get("workspace", {}).get("base_dir", "workspace")).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    steps: List[Dict[str, Any]] = []
    final_response = ""

    with _lock:
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

            if tool is None:
                final_response = assistant
                steps.append({"type": "response", "content": assistant})
                break

            if outside:
                steps.append({"type": "text", "content": outside})

            try:
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
                out = runner(payload) if runner else f"Bilinmeyen tool: {tool}"
            except Exception as e:
                out = f"Tool hatası: {e}"

            steps.append({"type": "tool", "tool": tool, "output": out[:3000]})

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
        "session_id": session_id,
        "total_steps": len(steps),
    })


# ─────────────────────────────────────────────
#  Veri Seti Kataloğu
# ─────────────────────────────────────────────

@app.route("/api/datasets", methods=["GET"])
def list_datasets():
    """
    Veri seti kataloğu.

    Query params:
        ?category=medical
        ?task_type=classification
    """
    _increment_requests()
    try:
        from dataset_catalog import (
            list_datasets as _list_ds,
            get_categories,
            format_catalog_for_prompt,
        )
        category = request.args.get("category")
        task_type = request.args.get("task_type")
        datasets = _list_ds(category=category, task_type=task_type)
        return jsonify({
            "datasets": datasets,
            "total": len(datasets),
            "categories": get_categories(),
        })
    except Exception as e:
        return _json_error(str(e), 500)


@app.route("/api/datasets/<dataset_id>/load", methods=["POST"])
def load_dataset(dataset_id: str):
    """Veri setini yükle — özellik, örnek, dağılım bilgileri döner."""
    _increment_requests()
    try:
        from dataset_catalog import load_dataset as _load_ds
        import numpy as np

        X, y, features = _load_ds(dataset_id)
        unique, counts = np.unique(y, return_counts=True)
        return jsonify({
            "dataset_id": dataset_id,
            "samples": int(X.shape[0]),
            "features": int(X.shape[1]),
            "feature_names": list(features),
            "target_unique": len(unique),
            "target_distribution": {str(k): int(v) for k, v in zip(unique, counts)},
        })
    except ValueError as e:
        return _json_error(str(e), 404)
    except Exception as e:
        return _json_error(str(e), 500)


@app.route("/api/datasets/search", methods=["GET"])
def search_datasets():
    """
    Veri seti ara.

    Query params:
        ?q=cancer
    """
    _increment_requests()
    try:
        from dataset_catalog import search_datasets as _search_ds
        query = request.args.get("q", "")
        results = _search_ds(query)
        return jsonify({"query": query, "results": results, "total": len(results)})
    except Exception as e:
        return _json_error(str(e), 500)


# ─────────────────────────────────────────────
#  Kaydedilmiş Modeller
# ─────────────────────────────────────────────

@app.route("/api/models", methods=["GET"])
def list_models():
    """Workspace ve results altındaki .pkl model dosyalarını listele."""
    _increment_requests()
    models = []
    search_dirs = [BASE_DIR / "workspace", BASE_DIR / "results"]
    for sdir in search_dirs:
        if not sdir.exists():
            continue
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


@app.route("/api/models/predict", methods=["POST"])
def predict():
    """
    Kaydedilmiş modelle tahmin yap.

    Body JSON:
        {
            "model_path": "workspace/project/results/best_model.pkl",
            "data": [[5.1, 3.5, 1.4, 0.2], [6.2, 2.9, 4.3, 1.3]]
        }
    """
    _increment_requests()
    body = request.get_json(force=True)
    model_path = body.get("model_path", "")
    data = body.get("data", [])

    if not model_path:
        return _json_error("'model_path' gerekli.")
    if not data:
        return _json_error("'data' boş olamaz.")

    try:
        from utils.model_loader import load_model
        import numpy as np

        full_path = str(BASE_DIR / model_path)
        model = load_model(full_path)
        X = np.array(data)
        predictions = model.predict(X).tolist()

        result = {"predictions": predictions, "count": len(predictions)}

        # Olasılık desteği
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X).tolist()
                result["probabilities"] = proba
            except Exception:
                pass

        return jsonify(result)
    except FileNotFoundError:
        return _json_error(f"Model dosyası bulunamadı: {model_path}", 404)
    except Exception as e:
        return _json_error(str(e), 500)


# ─────────────────────────────────────────────
#  LLM Backend'ler
# ─────────────────────────────────────────────

@app.route("/api/backends", methods=["GET"])
def list_backends():
    """Kullanılabilir LLM backend'leri listele."""
    _increment_requests()
    try:
        from llm_backend import list_backends as _list_be, create_backend
        backends = []
        for name in _list_be():
            try:
                b = create_backend(name)
                backends.append({
                    "name": name,
                    "available": b.is_available(),
                    "model": getattr(b, "model", ""),
                })
            except Exception:
                backends.append({"name": name, "available": False, "model": ""})
        return jsonify({"backends": backends})
    except Exception as e:
        return _json_error(str(e), 500)


# ─────────────────────────────────────────────
#  Yapılandırma
# ─────────────────────────────────────────────

@app.route("/api/config", methods=["GET"])
def get_config():
    """Mevcut yapılandırmayı döndür."""
    _increment_requests()
    cfg = _load_config()
    # API key'leri gizle
    safe_cfg = json.loads(json.dumps(cfg))
    for section in safe_cfg.values():
        if isinstance(section, dict):
            for key in list(section.keys()):
                if "key" in key.lower() or "token" in key.lower() or "secret" in key.lower():
                    section[key] = "***" if section[key] else None
    return jsonify(safe_cfg)


@app.route("/api/config", methods=["PUT"])
def update_config():
    """Yapılandırmayı güncelle."""
    _increment_requests()
    body = request.get_json(force=True)
    cfg = _load_config()

    def deep_update(base, updates):
        for k, v in updates.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                deep_update(base[k], v)
            else:
                base[k] = v

    deep_update(cfg, body)
    _save_config(cfg)
    return jsonify({"status": "ok", "message": "Yapılandırma güncellendi."})


# ─────────────────────────────────────────────
#  Proje İstatistikleri
# ─────────────────────────────────────────────

@app.route("/api/stats", methods=["GET"])
def stats():
    """Proje istatistikleri."""
    _increment_requests()
    modules = []
    py_files = list(BASE_DIR.glob("*.py")) + list((BASE_DIR / "utils").glob("*.py"))
    total_lines = 0
    for f in py_files:
        lines = len(f.read_text(encoding="utf-8", errors="replace").splitlines())
        total_lines += lines
        modules.append({"file": f.name, "lines": lines})

    test_files = list((BASE_DIR / "tests").glob("test_*.py"))
    test_count = 0
    for tf in test_files:
        content = tf.read_text(encoding="utf-8", errors="replace")
        test_count += content.count("def test_")

    return jsonify({
        "total_modules": len(modules),
        "total_lines": total_lines,
        "total_tests": test_count,
        "modules": sorted(modules, key=lambda m: m["lines"], reverse=True)[:10],
    })


# ═══════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Bio-ML Agent REST API Sunucusu",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Endpoint Örnekleri:
  curl http://localhost:8000/api/health
  curl http://localhost:8000/api/datasets
  curl http://localhost:8000/api/backends
  curl -X POST http://localhost:8000/api/chat \\
       -H "Content-Type: application/json" \\
       -d '{"message": "Breast cancer analizi yap"}'
        """,
    )
    parser.add_argument("--host", default="0.0.0.0", help="Dinlenecek adres (varsayılan: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port numarası (varsayılan: 8000)")
    parser.add_argument("--model", default="", help="Varsayılan LLM model adı")
    parser.add_argument("--debug", action="store_true", help="Debug modu")
    args = parser.parse_args()

    # Model config'e yaz
    if args.model:
        cfg = _load_config()
        cfg.setdefault("agent", {})["model"] = args.model
        _save_config(cfg)

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║   🔌 Bio-ML Agent — REST API                    ║")
    print(f"║   📍 http://{args.host}:{args.port}                     ║")
    print("║                                                  ║")
    print("║   Endpoint'ler:                                  ║")
    print("║     POST /api/chat        — Sohbet               ║")
    print("║     GET  /api/datasets    — Veri setleri          ║")
    print("║     GET  /api/models      — Modeller              ║")
    print("║     GET  /api/backends    — LLM backend'ler       ║")
    print("║     POST /api/models/predict — Tahmin             ║")
    print("║     GET  /api/config      — Ayarlar               ║")
    print("║     GET  /api/health      — Sağlık durumu         ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
