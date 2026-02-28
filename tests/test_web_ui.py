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

    def test_import_format_tool_output(self):
        from web_ui import _format_tool_output
        assert callable(_format_tool_output)

    def test_import_reset_session(self):
        from web_ui import _reset_session
        assert callable(_reset_session)

    def test_import_run_tool(self):
        from web_ui import _run_tool
        assert callable(_run_tool)

    def test_import_process_message(self):
        from web_ui import process_message
        assert callable(process_message)

    def test_import_create_ui(self):
        from web_ui import create_ui
        assert callable(create_ui)


# ─────────────────────────────────────────────
#  _format_tool_output Testleri
# ─────────────────────────────────────────────

class TestFormatToolOutput:
    """Tool çıktı formatlama testleri."""

    def test_python_output(self):
        from web_ui import _format_tool_output
        result = _format_tool_output("PYTHON", "Hello World")
        assert "🐍" in result
        assert "PYTHON" in result
        assert "Hello World" in result
        assert "```" in result

    def test_bash_output(self):
        from web_ui import _format_tool_output
        result = _format_tool_output("BASH", "total 42")
        assert "💻" in result
        assert "BASH" in result

    def test_web_search_json_output(self):
        from web_ui import _format_tool_output
        import json
        data = [
            {"title": "Test", "href": "https://example.com", "body": "Test body"},
        ]
        result = _format_tool_output("WEB_SEARCH", json.dumps(data))
        assert "🌐" in result
        assert "Test" in result

    def test_web_search_non_json(self):
        from web_ui import _format_tool_output
        result = _format_tool_output("WEB_SEARCH", "not json")
        assert "🌐" in result

    def test_read_file_output(self):
        from web_ui import _format_tool_output
        result = _format_tool_output("READ_FILE", "file contents here")
        assert "📄" in result

    def test_write_file_output(self):
        from web_ui import _format_tool_output
        result = _format_tool_output("WRITE_FILE", "written")
        assert "✍️" in result

    def test_todo_output(self):
        from web_ui import _format_tool_output
        result = _format_tool_output("TODO", "task added")
        assert "📝" in result

    def test_unknown_tool_output(self):
        from web_ui import _format_tool_output
        result = _format_tool_output("UNKNOWN_TOOL", "output")
        assert "🛠️" in result
        assert "output" in result


# ─────────────────────────────────────────────
#  _run_tool Testleri (Mock ile)
# ─────────────────────────────────────────────

class TestRunTool:
    """Tool çalıştırma testleri (mock ile)."""

    def _make_config(self, tmp_path):
        from web_ui import AgentConfig
        return AgentConfig(
            model="test",
            workspace=tmp_path,
            timeout=30,
            max_steps=5,
            history_dir=tmp_path / "history",
        )

    def test_unknown_tool(self, tmp_path):
        from web_ui import _run_tool
        cfg = self._make_config(tmp_path)
        result = _run_tool("NONEXISTENT", "payload", cfg, allow_web=False)
        assert "ERROR" in result
        assert "Bilinmeyen" in result

    @patch("web_ui.web_search")
    def test_web_search_blocked(self, mock_ws, tmp_path):
        from web_ui import _run_tool
        cfg = self._make_config(tmp_path)
        result = _run_tool("WEB_SEARCH", "query", cfg, allow_web=False)
        assert "BLOCKED" in result
        mock_ws.assert_not_called()

    @patch("web_ui.web_search", return_value="results")
    def test_web_search_allowed(self, mock_ws, tmp_path):
        from web_ui import _run_tool
        cfg = self._make_config(tmp_path)
        result = _run_tool("WEB_SEARCH", "query", cfg, allow_web=True)
        assert result == "results"
        mock_ws.assert_called_once()

    @patch("web_ui.read_file", return_value="file_content")
    def test_read_file(self, mock_rf, tmp_path):
        from web_ui import _run_tool
        cfg = self._make_config(tmp_path)
        result = _run_tool("READ_FILE", "test.py", cfg, allow_web=False)
        assert result == "file_content"

    @patch("web_ui.append_todo", return_value="ok")
    def test_todo(self, mock_todo, tmp_path):
        from web_ui import _run_tool
        cfg = self._make_config(tmp_path)
        result = _run_tool("TODO", "yeni görev", cfg, allow_web=False)
        assert result == "ok"


# ─────────────────────────────────────────────
#  _reset_session Testleri
# ─────────────────────────────────────────────

class TestResetSession:
    """Oturum yenileme testleri."""

    def test_reset_creates_session_id(self):
        from web_ui import _reset_session, _session_id
        _reset_session()
        import web_ui
        assert web_ui._session_id != ""
        assert len(web_ui._session_id) > 8

    def test_reset_initializes_messages(self):
        from web_ui import _reset_session
        _reset_session()
        import web_ui
        assert len(web_ui._messages) == 1
        assert web_ui._messages[0]["role"] == "system"

    def test_reset_creates_metadata(self):
        from web_ui import _reset_session
        _reset_session()
        import web_ui
        assert "created_at" in web_ui._session_metadata


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
