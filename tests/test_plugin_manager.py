# tests/test_plugin_manager.py
# ═══════════════════════════════════════════════════════════
#  Bio-ML Agent — Plugin Manager Test Suite
# ═══════════════════════════════════════════════════════════

import sys
import textwrap
from pathlib import Path

import pytest

# Proje kökünü path'e ekle
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from plugin_manager import PluginManager, ToolPlugin


# ─────────────────────────────────────────────
#  Test Plugin Fixtures
# ─────────────────────────────────────────────

class DummyPlugin(ToolPlugin):
    """Test için kullanılacak basit plugin."""

    @property
    def name(self) -> str:
        return "DUMMY"

    @property
    def description(self) -> str:
        return "Test amaçlı dummy plugin"

    def execute(self, payload: str, workspace: Path) -> str:
        return f"DUMMY_OUTPUT: {payload}"


class AnotherPlugin(ToolPlugin):
    """İkinci test plugin'i."""

    @property
    def name(self) -> str:
        return "ANOTHER"

    @property
    def description(self) -> str:
        return "İkinci test plugin"

    def execute(self, payload: str, workspace: Path) -> str:
        return f"ANOTHER: {payload.upper()}"


class ErrorPlugin(ToolPlugin):
    """Hata fırlatan plugin."""

    @property
    def name(self) -> str:
        return "ERROR_PLUGIN"

    @property
    def description(self) -> str:
        return "Hata fırlatan test plugin"

    def execute(self, payload: str, workspace: Path) -> str:
        raise RuntimeError("Bilerek hata fırlatıldı")


# ─────────────────────────────────────────────
#  ToolPlugin Base Class Tests
# ─────────────────────────────────────────────

class TestToolPlugin:
    """ToolPlugin abstract sınıf testleri."""

    def test_cannot_instantiate_abc(self):
        """ToolPlugin doğrudan instantiate edilemez."""
        with pytest.raises(TypeError):
            ToolPlugin()

    def test_dummy_plugin_name(self):
        p = DummyPlugin()
        assert p.name == "DUMMY"

    def test_dummy_plugin_description(self):
        p = DummyPlugin()
        assert "dummy" in p.description.lower() or "test" in p.description.lower()

    def test_execute_returns_string(self):
        p = DummyPlugin()
        result = p.execute("hello", Path("/tmp"))
        assert isinstance(result, str)
        assert "hello" in result

    def test_get_prompt_info(self):
        p = DummyPlugin()
        info = p.get_prompt_info()
        assert "DUMMY" in info
        assert p.description in info

    def test_repr(self):
        p = DummyPlugin()
        r = repr(p)
        assert "DUMMY" in r
        assert "ToolPlugin" in r


# ─────────────────────────────────────────────
#  PluginManager Tests
# ─────────────────────────────────────────────

class TestPluginManagerInit:
    """PluginManager başlatma testleri."""

    def test_init_empty(self):
        pm = PluginManager()
        assert len(pm) == 0
        assert pm.tool_names == []

    def test_register_plugin(self):
        pm = PluginManager()
        pm.register(DummyPlugin())
        assert len(pm) == 1
        assert "DUMMY" in pm.tool_names

    def test_register_multiple(self):
        pm = PluginManager()
        pm.register(DummyPlugin())
        pm.register(AnotherPlugin())
        assert len(pm) == 2
        assert "DUMMY" in pm.tool_names
        assert "ANOTHER" in pm.tool_names

    def test_register_non_plugin_raises(self):
        pm = PluginManager()
        with pytest.raises(TypeError):
            pm.register("not_a_plugin")

    def test_register_overwrites_existing(self):
        pm = PluginManager()
        pm.register(DummyPlugin())
        pm.register(DummyPlugin())  # Aynı isimle tekrar
        assert len(pm) == 1


class TestPluginManagerGet:
    """Plugin get/retrieve testleri."""

    def test_get_existing(self):
        pm = PluginManager()
        pm.register(DummyPlugin())
        plugin = pm.get("DUMMY")
        assert plugin is not None
        assert plugin.name == "DUMMY"

    def test_get_case_insensitive(self):
        pm = PluginManager()
        pm.register(DummyPlugin())
        assert pm.get("dummy") is not None
        assert pm.get("Dummy") is not None

    def test_get_nonexistent_returns_none(self):
        pm = PluginManager()
        assert pm.get("NONEXISTENT") is None


class TestPluginManagerExecute:
    """Plugin çalıştırma testleri."""

    def test_execute_success(self):
        pm = PluginManager()
        pm.register(DummyPlugin())
        result = pm.execute("DUMMY", "test_payload", Path("/tmp"))
        assert "DUMMY_OUTPUT" in result
        assert "test_payload" in result

    def test_execute_unknown_raises(self):
        pm = PluginManager()
        with pytest.raises(KeyError, match="Bilinmeyen"):
            pm.execute("UNKNOWN_TOOL", "payload", Path("/tmp"))

    def test_execute_multiple_plugins(self):
        pm = PluginManager()
        pm.register(DummyPlugin())
        pm.register(AnotherPlugin())
        r1 = pm.execute("DUMMY", "hello", Path("/tmp"))
        r2 = pm.execute("ANOTHER", "hello", Path("/tmp"))
        assert "DUMMY_OUTPUT" in r1
        assert "ANOTHER" in r2
        assert "HELLO" in r2  # AnotherPlugin upper() yapar


class TestPluginManagerListAndPrompt:
    """list_plugins ve get_prompt_additions testleri."""

    def test_list_plugins_empty(self):
        pm = PluginManager()
        assert pm.list_plugins() == []

    def test_list_plugins_with_entries(self):
        pm = PluginManager()
        pm.register(DummyPlugin())
        pm.register(AnotherPlugin())
        plugins = pm.list_plugins()
        assert len(plugins) == 2
        names = [p["name"] for p in plugins]
        assert "DUMMY" in names
        assert "ANOTHER" in names

    def test_list_plugins_has_description(self):
        pm = PluginManager()
        pm.register(DummyPlugin())
        plugins = pm.list_plugins()
        assert "description" in plugins[0]
        assert plugins[0]["description"] != ""

    def test_prompt_additions_empty(self):
        pm = PluginManager()
        assert pm.get_prompt_additions() == ""

    def test_prompt_additions_nonempty(self):
        pm = PluginManager()
        pm.register(DummyPlugin())
        additions = pm.get_prompt_additions()
        assert "DUMMY" in additions
        assert "EK TOOL" in additions


class TestPluginManagerDiscover:
    """Plugin keşif testleri (dosya sistemi ile)."""

    def test_discover_nonexistent_dir(self):
        pm = PluginManager()
        count = pm.discover("/tmp/nonexistent_plugin_dir_12345")
        assert count == 0

    def test_discover_empty_dir(self, tmp_path):
        pm = PluginManager()
        count = pm.discover(str(tmp_path))
        assert count == 0

    def test_discover_skips_underscore_files(self, tmp_path):
        (tmp_path / "__init__.py").write_text("# init")
        (tmp_path / "_private.py").write_text("# private")
        pm = PluginManager()
        count = pm.discover(str(tmp_path))
        assert count == 0

    def test_discover_loads_plugin_file(self, tmp_path):
        """Geçerli bir plugin dosyasını keşfeder ve yükler."""
        plugin_code = textwrap.dedent(f"""\
            import sys
            sys.path.insert(0, r'{ROOT}')
            from pathlib import Path
            from plugin_manager import ToolPlugin

            class TestDiscoverPlugin(ToolPlugin):
                @property
                def name(self): return "DISCOVER_TEST"
                @property
                def description(self): return "Discover test"
                def execute(self, payload, workspace):
                    return "discovered"
        """)
        (tmp_path / "test_discover_plugin.py").write_text(plugin_code)
        pm = PluginManager()
        count = pm.discover(str(tmp_path))
        assert count == 1
        assert pm.get("DISCOVER_TEST") is not None

    def test_discover_real_plugins_dir(self):
        """Mevcut plugins/ klasöründen en az 1 plugin yüklenir."""
        plugins_dir = ROOT / "plugins"
        if not plugins_dir.exists():
            pytest.skip("plugins/ dizini bulunamadı")
        pm = PluginManager()
        count = pm.discover(str(plugins_dir))
        assert count >= 1  # En az example_plugin.py'den LISTDIR + TREE
        assert len(pm) >= 1
