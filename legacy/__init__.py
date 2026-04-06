"""
legacy package — adds its own directory to sys.path so that the legacy
module files (agent.py, rag_engine.py, etc.) can use their original
bare imports (e.g. ``from llm_backend import …``).
"""
import sys
from pathlib import Path

_legacy_dir = str(Path(__file__).resolve().parent)
if _legacy_dir not in sys.path:
    sys.path.insert(0, _legacy_dir)
