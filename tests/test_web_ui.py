# tests/test_web_ui.py
# ═══════════════════════════════════════════════════════════
#  Bio-ML Agent — Web UI (Gradio) Test Suite
# ═══════════════════════════════════════════════════════════

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────
#  Import Testleri
# ─────────────────────────────────────────────

class TestWebUIImports:
    """Web UI modülünün import edilebilirliğini test eder."""

    def test_import_process_message(self):
        from web_ui import process_message
        assert callable(process_message)

    def test_import_create_ui(self):
        from web_ui import create_ui
        assert callable(create_ui)





# ─────────────────────────────────────────────
#  create_ui Testleri
# ─────────────────────────────────────────────

class TestCreateUI:
    """Gradio arayüz oluşturma testleri."""

    def test_create_ui_returns_blocks(self):
        """create_ui() çağrıldığında Gradio Blocks döndürür."""
        try:
            import gradio as gr
        except ImportError:
            pytest.skip("Gradio kurulu değil")

        from web_ui import create_ui
        demo = create_ui()
        assert isinstance(demo, gr.Blocks)

    def test_create_ui_has_title(self):
        """UI başlığı Bio-ML Agent içerir."""
        try:
            import gradio as gr
        except ImportError:
            pytest.skip("Gradio kurulu değil")

        from web_ui import create_ui
        demo = create_ui()
        assert demo.title is not None
        assert "Bio-ML" in demo.title

    def test_create_ui_has_theme_and_css(self):
        """create_ui() bir tema ve CSS ayarlar."""
        try:
            import gradio as gr
        except ImportError:
            pytest.skip("Gradio kurulu değil")

        from web_ui import create_ui
        demo = create_ui()
        assert hasattr(demo, "_bio_theme")
        assert hasattr(demo, "_bio_css")
        assert demo._bio_css != ""
