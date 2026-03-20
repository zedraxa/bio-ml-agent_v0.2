#!/usr/bin/env python3
# agent.py
# ═══════════════════════════════════════════════════════════
#  Bio-ML Agent — Ana CLI Giriş Noktası
#  Kullanım:
#    bio-ml-agent ui      → Gradio Web Arayüzünü başlatır
#    bio-ml-agent api     → FastAPI REST sunucusunu başlatır
#    bio-ml-agent chat    → Terminal'de interaktif sohbet
#    bio-ml-agent check   → Kurulum doğrulaması
# ═══════════════════════════════════════════════════════════

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Proje kökünü path'e ekle
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ─────────────────────────────────────────────
#  Alt Komutlar
# ─────────────────────────────────────────────

def cmd_ui(args: argparse.Namespace) -> None:
    """Gradio Web Arayüzünü başlat."""
    from bio_ml_agent.web_ui import main as ui_main
    print(f"🧠 Bio-ML Agent — Web UI başlatılıyor (port {args.port})...")
    ui_main()


def cmd_api(args: argparse.Namespace) -> None:
    """FastAPI REST API sunucusunu başlat."""
    import uvicorn
    print(f"🚀 Bio-ML Agent — API sunucusu başlatılıyor (port {args.port})...")
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def cmd_chat(args: argparse.Namespace) -> None:
    """Terminal'de interaktif sohbet oturumu."""
    from dotenv import load_dotenv
    load_dotenv()

    from bio_ml_agent.utils.config import load_config
    from bio_ml_agent.utils.logger import setup_logger
    from bio_ml_agent.services.agent_service import AgentService

    log_dir = Path("logs").resolve()
    log_dir.mkdir(exist_ok=True)
    setup_logger(log_dir, "WARNING")

    app_config = load_config()
    model = args.model or app_config.agent.model

    service = AgentService(
        model=model,
        timeout=app_config.agent.timeout,
        max_steps=app_config.agent.max_steps,
    )

    print("═" * 50)
    print("🧠 Bio-ML Agent — İnteraktif Sohbet")
    print(f"   Model : {model}")
    print(f"   Çıkış : 'q', 'exit' veya Ctrl+C")
    print("═" * 50)
    print()

    try:
        while True:
            try:
                user_msg = input("Sen > ").strip()
            except EOFError:
                break
            if not user_msg:
                continue
            if user_msg.lower() in ("q", "exit", "quit", "çık", "çıkış"):
                break

            print()
            for event in service.process_message(user_msg):
                ev_type = event.get("type")
                if ev_type == "status":
                    print(f"  ⏳ {event.get('content', '')}")
                elif ev_type in ("chunk",):
                    print(event.get("content", ""), end="", flush=True)
                elif ev_type == "assistant":
                    content = event.get("content", "")
                    if content:
                        print(content)
                elif ev_type == "tool_start":
                    print(f"\n  🔧 [{event.get('tool', '')}] çalışıyor...")
                elif ev_type == "tool_output":
                    out = event.get("output", "")
                    # Uzun çıktıları kısalt
                    if len(out) > 500:
                        out = out[:500] + f"\n  ... ({len(out)} karakter, kısaltıldı)"
                    print(f"  📋 Çıktı:\n{out}")
                elif ev_type == "error":
                    print(f"\n  ❌ {event.get('content', '')}")
                elif ev_type == "done":
                    pass
            print()

    except KeyboardInterrupt:
        print("\n\n👋 Görüşürüz!")


def cmd_check(args: argparse.Namespace) -> None:
    """Kurulum doğrulaması — tüm kritik modülleri kontrol et."""
    print("🔍 Bio-ML Agent — Kurulum Kontrolü")
    print("═" * 50)

    checks = [
        ("core.agent_core",       "AgentCore"),
        ("core.config",           "Config"),
        ("core.tools",            "Tools"),
        ("core.conversation",     "Conversation"),
        ("models",                "Models (Pydantic)"),
        ("services.agent_service","AgentService"),
        ("llm_backend",           "LLM Backend"),
        ("ml",                    "ML Package"),
        ("exceptions",            "Exceptions"),
        ("plugin_manager",        "Plugin Manager"),
        ("ultra_agent",           "Ultra Agent"),
    ]

    ok = 0
    fail = 0
    for mod, label in checks:
        try:
            __import__(mod)
            print(f"  ✅ {label:<25} ({mod})")
            ok += 1
        except Exception as e:
            print(f"  ❌ {label:<25} ({mod}): {e}")
            fail += 1

    print("═" * 50)
    print(f"  Sonuç: {ok} başarılı, {fail} başarısız")

    if fail > 0:
        sys.exit(1)


# ─────────────────────────────────────────────
#  argparse Yapılandırması
# ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bio-ml-agent",
        description="🧠 Bio-ML Agent — Yerel LLM destekli Biyomühendislik ML Proje Asistanı",
    )
    sub = parser.add_subparsers(dest="command")

    # ui
    p_ui = sub.add_parser("ui", help="Gradio Web Arayüzünü başlat (varsayılan)")
    p_ui.add_argument("--port", type=int, default=7860, help="Dinlenecek port (varsayılan: 7860)")

    # api
    p_api = sub.add_parser("api", help="FastAPI REST API sunucusunu başlat")
    p_api.add_argument("--host", default="0.0.0.0", help="Dinlenecek adres (varsayılan: 0.0.0.0)")
    p_api.add_argument("--port", type=int, default=8001, help="Dinlenecek port (varsayılan: 8001)")
    p_api.add_argument("--reload", action="store_true", help="Geliştirme modunda auto-reload")

    # chat
    p_chat = sub.add_parser("chat", help="Terminal'de interaktif sohbet")
    p_chat.add_argument("--model", default="", help="Kullanılacak model (varsayılan: config'den)")

    # check
    sub.add_parser("check", help="Kurulum doğrulaması")

    return parser


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "ui":    cmd_ui,
        "api":   cmd_api,
        "chat":  cmd_chat,
        "check": cmd_check,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        # Argüman verilmezse varsayılan: UI
        args.port = 7860
        cmd_ui(args)
    else:
        handler(args)


if __name__ == "__main__":
    main()
