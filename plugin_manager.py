# plugin_manager.py
# ═══════════════════════════════════════════════════════════
#  Bio-ML Agent — Plugin Sistemi
#  Dinamik tool yükleme ve yönetimi.
# ═══════════════════════════════════════════════════════════

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("bio_ml_agent")


# ─────────────────────────────────────────────
#  Tool Plugin Base Class
# ─────────────────────────────────────────────

class ToolPlugin(ABC):
    """Tüm tool plugin'lerinin temel sınıfı.

    Yeni bir tool eklemek için:
        1. Bu sınıftan türetin.
        2. `name`, `description` özelliklerini tanımlayın.
        3. `execute()` metodunu uygulayın.
        4. Dosyayı `plugins/` klasörüne koyun.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool'un büyük harfli adı (ör: 'IMAGE', 'LISTDIR')."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool'un kısa açıklaması."""
        ...

    @abstractmethod
    def execute(self, payload: str, workspace: Path) -> str:
        """Tool payload'ını çalıştır ve sonucu döndür.

        Args:
            payload: Tool bloğunun içeriği.
            workspace: Agent'ın çalışma dizini.

        Returns:
            Tool çıktısı (str).
        """
        ...

    def get_prompt_info(self) -> str:
        """System prompt'a eklenecek tool açıklaması."""
        return f"<{self.name}>...</{self.name}> — {self.description}"

    def __repr__(self) -> str:
        return f"<ToolPlugin name={self.name!r}>"


# ─────────────────────────────────────────────
#  Plugin Manager
# ─────────────────────────────────────────────

class PluginManager:
    """Plugin'leri keşfeder, yükler ve yönetir.

    Kullanım:
        pm = PluginManager()
        pm.discover("plugins/")
        tool = pm.get("LISTDIR")
        result = tool.execute(payload, workspace)
    """

    def __init__(self):
        self._plugins: Dict[str, ToolPlugin] = {}

    def register(self, plugin: ToolPlugin) -> None:
        """Bir plugin'i kaydet.

        Args:
            plugin: ToolPlugin instance.
        """
        if not isinstance(plugin, ToolPlugin):
            raise TypeError(f"{plugin!r} ToolPlugin alt sınıfı olmalıdır.")
        name = plugin.name.upper()
        if name in self._plugins:
            log.warning("⚠️ Plugin üzerine yazılıyor: %s", name)
        self._plugins[name] = plugin
        log.info("🔌 Plugin kaydedildi: %s — %s", name, plugin.description)

    def discover(self, plugin_dir: str | Path) -> int:
        """Bir klasördeki tüm plugin'leri otomatik keşfet ve yükle.

        Her .py dosyası taranır; ToolPlugin alt sınıfları bulunur ve kaydedilir.

        Args:
            plugin_dir: Plugin dosyalarının bulunduğu klasör.

        Returns:
            Yüklenen plugin sayısı.
        """
        plugin_path = Path(plugin_dir)
        if not plugin_path.is_dir():
            log.warning("⚠️ Plugin dizini bulunamadı: %s", plugin_path)
            return 0

        count = 0
        for py_file in sorted(plugin_path.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            try:
                loaded = self._load_module(py_file)
                count += loaded
            except Exception as e:
                log.error("❌ Plugin yüklenemedi: %s | %s", py_file.name, e)
        return count

    def _load_module(self, filepath: Path) -> int:
        """Bir Python dosyasından ToolPlugin alt sınıflarını yükle."""
        module_name = f"plugin_{filepath.stem}"
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        if spec is None or spec.loader is None:
            return 0

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        count = 0
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, ToolPlugin)
                and attr is not ToolPlugin
            ):
                try:
                    instance = attr()
                    self.register(instance)
                    count += 1
                except Exception as e:
                    log.error("❌ Plugin örneği oluşturulamadı: %s | %s", attr_name, e)
        return count

    def get(self, name: str) -> Optional[ToolPlugin]:
        """İsme göre plugin getir."""
        return self._plugins.get(name.upper())

    def list_plugins(self) -> List[Dict[str, str]]:
        """Kayıtlı plugin'lerin listesini döndür."""
        return [
            {"name": p.name, "description": p.description}
            for p in self._plugins.values()
        ]

    @property
    def tool_names(self) -> List[str]:
        """Kayıtlı tool isimlerini döndür."""
        return list(self._plugins.keys())

    def execute(self, tool_name: str, payload: str, workspace: Path) -> str:
        """Tool'u çalıştır.

        Args:
            tool_name: Tool adı.
            payload: Tool payload'ı.
            workspace: Çalışma dizini.

        Returns:
            Tool çıktısı.

        Raises:
            KeyError: Bilinmeyen tool.
        """
        plugin = self.get(tool_name)
        if plugin is None:
            raise KeyError(f"Bilinmeyen plugin: {tool_name}")
        log.info("🔌 Plugin çalıştırılıyor: %s", tool_name)
        return plugin.execute(payload, workspace)

    def get_prompt_additions(self) -> str:
        """Tüm plugin'lerin system prompt ek bilgilerini oluştur."""
        if not self._plugins:
            return ""
        lines = ["\n\nEK TOOL'LAR (Plugin):"]
        for p in self._plugins.values():
            lines.append(p.get_prompt_info())
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._plugins)
