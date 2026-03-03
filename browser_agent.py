"""
Browser Sub-Agent — Otonom Headless Browser Kontrolü
Agent'a bir görev ver, o kendi başına tarayıcıyı açıp düşünerek hareket eder.
Antigravity'nin browser_subagent'ı gibi çalışır.
"""

import re
import time
import json
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger("browser_agent")


BROWSER_AGENT_SYSTEM = """Sen bir Browser Sub-Agent'sın. Headless bir tarayıcıyı kontrol ederek görevleri yerine getirirsin.

Her adımda sana:
1. Görev açıklaması
2. Şu anki sayfa durumu (title, url, görünür metin)
3. Önceki adımların geçmişi

verilir. Sen şu komutlardan BİRİNİ seç ve döndür:

KOMUTLAR:
- goto: URL → Sayfaya git
- click: CSS_SELECTOR → Elemente tıkla
- type: CSS_SELECTOR | metin → Elemente yaz
- press: Enter/Tab/Escape → Tuşa bas
- scroll: down/up/bottom/top → Scroll
- wait: N → N saniye bekle
- text: CSS_SELECTOR → Elementin textini oku (body için: text: body)
- screenshot: dosya.png → Ekran görüntüsü al
- done: SONUÇ → Görev tamamlandı, sonucu bildir

KURALLAR:
- Her yanıtında SADECE bir komut ver
- Komutu şu formatta ver: <CMD>komut: argüman</CMD>
- Tıklamadan önce doğru selektörü kullan
- Sayfanın yüklenmesi için gerekirse wait kullan
- Görev tamamlandığında MUTLAKA done: ile sonucu bildir
- Maksimum 15 adımda görevi bitir

Örnek yanıt:
Sayfa yüklendi, arama kutusuna yazıyorum.
<CMD>type: input[name=q] | playwright nedir</CMD>
"""


class BrowserSubAgent:
    """LLM-powered otonom browser agent."""

    def __init__(self, model: str = None, workspace: Path = None, max_steps: int = 15):
        self.model = model
        self.workspace = workspace or Path(".")
        self.max_steps = max_steps
        self.history = []

    def execute(self, task: str) -> str:
        """Görevi otonom olarak tarayıcıda çalıştır."""
        log.info("🌐 BrowserSubAgent başlatıldı | görev='%s'", task[:100])

        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return "[BROWSER_AGENT HATA] Playwright yüklü değil. Kur: pip install playwright && playwright install chromium"

        # LLM backend'i al
        try:
            from llm_backend import call_llm
        except ImportError:
            return "[BROWSER_AGENT HATA] LLM backend import edilemedi."

        results = []
        start_time = time.time()

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(
                    user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
                    viewport={"width": 1280, "height": 720},
                )

                messages = [
                    {"role": "system", "content": BROWSER_AGENT_SYSTEM},
                ]

                for step in range(1, self.max_steps + 1):
                    # Mevcut sayfa durumunu al
                    try:
                        page_title = page.title() or "(boş)"
                        page_url = page.url or "about:blank"
                        # Body textinin ilk 2000 karakterini al
                        try:
                            page_text = page.inner_text("body")
                            page_text = re.sub(r"\n{3,}", "\n\n", page_text).strip()
                            page_text = page_text[:2000]
                        except Exception:
                            page_text = "(sayfa içeriği okunamadı)"
                    except Exception:
                        page_title = "(bilinmiyor)"
                        page_url = "about:blank"
                        page_text = "(sayfa yok)"

                    # Kullanıcı mesajını oluştur
                    step_prompt = f"""Adım {step}/{self.max_steps}

GÖREV: {task}

SAYFA DURUMU:
- Title: {page_title}
- URL: {page_url}
- İçerik (ilk 2000 karakter):
{page_text}

GEÇMIŞ ADIMLAR:
{chr(10).join(results[-5:]) if results else "(henüz adım yok)"}

Şimdi ne yapmalısın? Bir sonraki komutu ver."""

                    messages.append({"role": "user", "content": step_prompt})

                    # LLM'den yanıt al
                    try:
                        llm_response = call_llm(
                            messages,
                            model=self.model,
                            timeout=30,
                        )
                    except Exception as e:
                        results.append(f"[{step}] ❌ LLM hatası: {e}")
                        break

                    messages.append({"role": "assistant", "content": llm_response})

                    # Komutu parse et
                    cmd_match = re.search(r"<CMD>\s*(.*?)\s*</CMD>", llm_response, re.DOTALL)
                    if not cmd_match:
                        results.append(f"[{step}] ⚠️ Komut bulunamadı: {llm_response[:100]}")
                        continue

                    raw_cmd = cmd_match.group(1).strip()
                    cmd, _, arg = raw_cmd.partition(":")
                    cmd = cmd.strip().lower()
                    arg = arg.strip()

                    log.info("🌐 BrowserSubAgent adım %d | %s: %s", step, cmd, arg[:80])

                    # Komutu çalıştır
                    try:
                        if cmd == "goto":
                            page.goto(arg, wait_until="domcontentloaded", timeout=30000)
                            page.wait_for_timeout(1500)
                            results.append(f"[{step}] goto: {arg} → ✅")

                        elif cmd == "click":
                            page.click(arg, timeout=10000)
                            page.wait_for_timeout(1000)
                            results.append(f"[{step}] click: {arg} → ✅")

                        elif cmd == "type":
                            parts = arg.split("|", 1)
                            if len(parts) == 2:
                                selector, text = parts[0].strip(), parts[1].strip()
                                page.fill(selector, text)
                                results.append(f"[{step}] type: {selector} → '{text[:50]}' ✅")
                            else:
                                results.append(f"[{step}] type: ❌ Format: CSS_SELECTOR | metin")

                        elif cmd == "press":
                            page.keyboard.press(arg)
                            page.wait_for_timeout(500)
                            results.append(f"[{step}] press: {arg} → ✅")

                        elif cmd == "scroll":
                            direction = arg.lower()
                            if direction == "down":
                                page.evaluate("window.scrollBy(0, 500)")
                            elif direction == "up":
                                page.evaluate("window.scrollBy(0, -500)")
                            elif direction == "bottom":
                                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                            elif direction == "top":
                                page.evaluate("window.scrollTo(0, 0)")
                            page.wait_for_timeout(300)
                            results.append(f"[{step}] scroll: {direction} → ✅")

                        elif cmd == "wait":
                            secs = min(float(arg) if arg else 1, 10)
                            page.wait_for_timeout(int(secs * 1000))
                            results.append(f"[{step}] wait: {secs}s → ✅")

                        elif cmd == "text":
                            selector = arg or "body"
                            text_content = page.inner_text(selector)
                            text_content = re.sub(r"\n{3,}", "\n\n", text_content).strip()
                            if len(text_content) > 3000:
                                text_content = text_content[:3000] + "\n[TRUNCATED]"
                            results.append(f"[{step}] text ({selector}):\n{text_content}")

                        elif cmd == "screenshot":
                            filename = arg or f"screenshot_{step}.png"
                            save_path = self.workspace / filename
                            save_path.parent.mkdir(parents=True, exist_ok=True)
                            page.screenshot(path=str(save_path), full_page=False)
                            results.append(f"[{step}] screenshot: {save_path} → ✅")

                        elif cmd == "done":
                            results.append(f"[{step}] ✅ GÖREV TAMAMLANDI: {arg}")
                            break

                        else:
                            results.append(f"[{step}] ❌ Bilinmeyen komut: {cmd}")

                    except Exception as e:
                        results.append(f"[{step}] ❌ {cmd}: {type(e).__name__}: {str(e)[:200]}")

                browser.close()

        except Exception as e:
            log.error("🌐 BrowserSubAgent HATA: %s", e, exc_info=True)
            results.append(f"\n❌ Browser hatası: {type(e).__name__}: {str(e)[:300]}")

        elapsed = time.time() - start_time
        log.info("🌐 BrowserSubAgent tamamlandı | süre=%.2fs | adım=%d", elapsed, len(results))

        output = "\n".join(results)
        return f"[BROWSER_AGENT] {len(results)} adım | {elapsed:.1f}s\n\n{output}"


def run_browser_agent(task: str, model: str = None, workspace: Path = None) -> str:
    """Kolaylık fonksiyonu — BrowserSubAgent'ı çalıştır."""
    agent = BrowserSubAgent(model=model, workspace=workspace)
    return agent.execute(task)
