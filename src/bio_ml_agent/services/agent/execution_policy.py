# services/agent/execution_policy.py
import logging
from typing import Set, Optional

log = logging.getLogger("bio_ml_agent")

_ACTION_KEYWORDS = [
    "oluştur", "olustur", "yaz", "kur", "indir", "kaydet", "eğit", "egit",
    "temizle", "analiz", "karşılaştır", "karsilastir", "grafik", "rapor",
    "create", "build", "download", "save", "train", "write", "generate",
    "proje", "project", "dosya", "file", "model", "plot", "report",
    "pipeline", "csv", "dataset", "veri set",
]

TOOL_ENFORCEMENT_PROMPT = (
    "UYARI: Kullanıcı dosya oluşturma, veri indirme veya kod çalıştırma istedi "
    "ama sen hiç tool çağrısı yapmadın. Bu KABUL EDİLEMEZ.\n\n"
    "ŞİMDİ şu tool'lardan birini MUTLAKA kullan:\n"
    "- <WRITE_FILE> ile dosya oluştur (plan.md veya proje dosyası)\n"
    "- <PYTHON> ile kod çalıştır\n"
    "- <BASH> ile komut çalıştır (curl/wget ile veri indir)\n"
    "- <WEB_SEARCH> ile veri kaynağı ara\n\n"
    "İlk adım olarak bir plan.md dosyası yaz, sonra her adımı sırayla tool ile uygula.\n"
    "ASLA düz metin açıklama yapma — tool çağrısı ZORUNLUDUR."
)

def is_action_request(msg: str) -> bool:
    """Mesajın dosya/proje oluşturma gibi aksiyon gerektiren bir istek olup olmadığını algılar."""
    msg_lower = msg.lower()
    hits = sum(1 for kw in _ACTION_KEYWORDS if kw in msg_lower)
    return hits >= 2

def needs_approval(step: int, mode: int, interval: int, checkpoint: int, plan_approved: bool, tool: Optional[str] = None) -> bool:
    """Mevcut adımda kullanıcı onayı gerekip gerekmediğini kontrol eder."""
    if mode == 1:  # Tam Otomatik
        return False
    elif mode == 2:  # Adım bazlı onay
        return step > 0 and step % interval == 0
    elif mode == 3:  # Akıllı onay (sadece kritik tool'larda)
        critical_tools = {"BASH", "WRITE_FILE", "WEB_SEARCH", "WEB_OPEN"}
        return tool in critical_tools
    elif mode == 4:  # Plan onayı + checkpoint
        if step == 1 and not plan_approved:
            return True
        if step > 0 and step == checkpoint:
            return True
        return False
    return False

def format_tool_output(tool: str, output: str) -> str:
    """Arayüz dökümleri (Markdown) için aracı çıktılarını şekillendirir."""
    import json
    icon_map = {
        "PYTHON": "🐍", "BASH": "💻", "WEB_SEARCH": "🌐",
        "WEB_OPEN": "📖", "READ_FILE": "📄", "WRITE_FILE": "✍️", "TODO": "📝"
    }
    icon = icon_map.get(tool, "🛠️")

    if tool in {"PYTHON", "BASH"}:
        return f"**{icon} {tool} Çıktısı:**\n```\n{output}\n```"
    elif tool == "WEB_SEARCH":
        try:
            results = json.loads(output)
            lines = [f"**{icon} Web Arama Sonuçları:**\n"]
            for r in results[:5]:
                lines.append(f"- [{r.get('title', 'N/A')}]({r.get('href', '#')})")
                lines.append(f"  _{r.get('body', '')[:120]}_\n")
            return "\n".join(lines)
        except:
            return f"**{icon} Web Arama:**\n```\n{output}\n```"
    else:
        return f"**{icon} {tool}:**\n```\n{output}\n```"
