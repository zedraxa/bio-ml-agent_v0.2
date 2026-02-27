# plugins/example_plugin.py
# ═══════════════════════════════════════════════════════════
#  Örnek Plugin: LISTDIR — Dizin içeriğini listeler
# ═══════════════════════════════════════════════════════════

from pathlib import Path

import sys
import os

# Proje kökünü path'e ekle (plugin_manager'dan import için)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plugin_manager import ToolPlugin


class ListDirPlugin(ToolPlugin):
    """Workspace içindeki bir dizinin dosyalarını listeler.

    Kullanım:
        <LISTDIR>src/</LISTDIR>
    """

    @property
    def name(self) -> str:
        return "LISTDIR"

    @property
    def description(self) -> str:
        return "Dizin içeriğini listeler (dosya ve klasörler)"

    def execute(self, payload: str, workspace: Path) -> str:
        target = payload.strip() or "."
        target_path = workspace / target

        if not target_path.exists():
            return f"[ERROR] Dizin bulunamadı: {target}"
        if not target_path.is_dir():
            return f"[ERROR] Bu bir dizin değil: {target}"

        items = []
        for item in sorted(target_path.iterdir()):
            if item.name.startswith("."):
                continue
            icon = "📁" if item.is_dir() else "📄"
            size = ""
            if item.is_file():
                kb = item.stat().st_size / 1024
                size = f" ({kb:.1f} KB)"
            items.append(f"  {icon} {item.name}{size}")

        if not items:
            return f"📁 {target} (boş dizin)"

        header = f"📁 {target} ({len(items)} öğe):"
        return header + "\n" + "\n".join(items)


class TreePlugin(ToolPlugin):
    """Workspace'in ağaç yapısını gösterir.

    Kullanım:
        <TREE>.</TREE>      (kök dizin)
        <TREE>src/</TREE>   (belirli dizin)
    """

    @property
    def name(self) -> str:
        return "TREE"

    @property
    def description(self) -> str:
        return "Dizin ağacını gösterir (maksimum 3 seviye)"

    def execute(self, payload: str, workspace: Path) -> str:
        target = payload.strip() or "."
        target_path = workspace / target

        if not target_path.exists():
            return f"[ERROR] Dizin bulunamadı: {target}"

        lines = [f"📁 {target}"]
        self._build_tree(target_path, lines, prefix="", max_depth=3, current_depth=0)
        return "\n".join(lines)

    def _build_tree(
        self, path: Path, lines: list, prefix: str,
        max_depth: int, current_depth: int
    ) -> None:
        if current_depth >= max_depth:
            return

        items = sorted(
            [i for i in path.iterdir() if not i.name.startswith(".")],
            key=lambda x: (not x.is_dir(), x.name),
        )

        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            connector = "└── " if is_last else "├── "
            icon = "📁" if item.is_dir() else "📄"
            lines.append(f"{prefix}{connector}{icon} {item.name}")

            if item.is_dir():
                extension = "    " if is_last else "│   "
                self._build_tree(
                    item, lines, prefix + extension,
                    max_depth, current_depth + 1,
                )
