"""
bio_ml_agent.agent — Unified CLI entry point.

Usage:
    bio-ml-agent api          Start the FastAPI server
    bio-ml-agent ui           Start the Gradio web UI
    bio-ml-agent worker       Start the background job worker
    bio-ml-agent gateway      Start the API gateway
    bio-ml-agent whatsapp     Start the WhatsApp connector
    bio-ml-agent --version    Show version
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="bio-ml-agent",
        description="Bio-ML Agent — Autonomous Bioengineering Lab Assistant",
    )
    parser.add_argument(
        "--version", action="store_true", help="Show version and exit"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("api", help="Start the FastAPI REST API server")
    subparsers.add_parser("ui", help="Start the Gradio web UI")
    subparsers.add_parser("worker", help="Start the background job worker")
    subparsers.add_parser("gateway", help="Start the API gateway server")
    subparsers.add_parser("whatsapp", help="Start the WhatsApp connector")

    args = parser.parse_args()

    if args.version:
        from importlib.metadata import version as pkg_version

        print(f"bio-ml-agent {pkg_version('bio-ml-agent')}")
        return

    if args.command == "api":
        from bio_ml_agent.api.api_server import main as api_main
        api_main()
    elif args.command == "ui":
        from bio_ml_agent.web_ui import main as ui_main
        ui_main()
    elif args.command == "worker":
        from bio_ml_agent.workers.job_worker import main as worker_main
        worker_main()
    elif args.command == "gateway":
        from bio_ml_agent.api.gateway_server import main as gateway_main
        gateway_main()
    elif args.command == "whatsapp":
        from bio_ml_agent.services.whatsapp_connector import main as wa_main
        wa_main()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
