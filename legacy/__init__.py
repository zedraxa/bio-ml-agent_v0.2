"""
legacy package — adds necessary directories to sys.path so that the legacy
module files (agent.py, rag_engine.py, etc.) can use their original
bare imports (e.g. ``from llm_backend import …``).
"""
import sys
from pathlib import Path

_legacy_dir = Path(__file__).resolve().parent
_repo_root = _legacy_dir.parent  # legacy/ is one level below repo root

if str(_legacy_dir) not in sys.path:
    sys.path.insert(0, str(_legacy_dir))

# Add src/bio_ml_agent so bare imports like ``from llm_backend import …``
# resolve to the canonical modules.
_src_bio = _repo_root / "src" / "bio_ml_agent"
if _src_bio.is_dir() and str(_src_bio) not in sys.path:
    sys.path.insert(0, str(_src_bio))

# Also add src/bio_ml_agent/legacy for nested legacy imports
_src_legacy = _src_bio / "legacy"
if _src_legacy.is_dir() and str(_src_legacy) not in sys.path:
    sys.path.insert(0, str(_src_legacy))
