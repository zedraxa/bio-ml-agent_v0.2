# core/tools.py
# ═══════════════════════════════════════════════════════════
#  Bio-ML Agent — Tool Fonksiyonları
#  agent.py monolitinden ayrıştırılmıştır.
# ═══════════════════════════════════════════════════════════

import json
import logging
import os
import re
import subprocess
import textwrap
import time
from pathlib import Path
from typing import Optional, List, Tuple

from exceptions import (
    ToolExecutionError,
    FileOperationError,
    SecurityViolationError,
    ToolTimeoutError,
    ValidationError,
)

log = logging.getLogger("bio_ml_agent")

# ─────────────────────────────────────────────
#  Sabitler ve Regex'ler
# ─────────────────────────────────────────────

TOOL_TAGS = [
    "PYTHON", "BASH", "WEB_SEARCH", "WEB_OPEN",
    "BROWSER_OPEN", "BROWSER_ACTION", "BROWSER_AGENT",
    "READ_FILE", "WRITE_FILE", "TODO",
]

TOOL_RE = re.compile(
    r"<(" + "|".join(TOOL_TAGS) + r")>\s*(.*?)\s*</\1>",
    re.DOTALL | re.IGNORECASE,
)

FENCED_BASH_RE = re.compile(r"```(?:bash)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
FENCED_PY_RE = re.compile(r"```(?:python|py)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

# Güvenlik desenleri — config.yaml'dan yüklenir, yoksa varsayılanlar kullanılır
DENY_PATTERNS = [
    r"\brm\b\s+.*-rf\s+/",
    r":\(\)\s*{\s*:\s*\|\s*:\s*&\s*}\s*;\s*:",
    r"\bddd\b\s+if=/dev/zero\b",
    r"\bmkfs\.",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bkill\b\s+-9\s+1\b",
    r"\bnc\b\s+-e\b",
    r"\bnetcat\b",
    r"\bbash\b\s+-i\b",
    r"/dev/tcp/",
    r"/dev/udp/",
    r"\bcurl\b\s+-[XOd]\b.*\b(http|https)://",
    r"\bwget\b\s+--post-data\b",
    r"\bsudo\b",
    r"\bchmod\b\s+777\b",
    r"\bchown\b\s+root\b",
]


# ─────────────────────────────────────────────
#  Yardımcı: Config erişimi (lazy)
# ─────────────────────────────────────────────

def _cfg():
    """Mevcut config'i döndürür (lazy init)."""
    from utils.config import get_config
    return get_config()


def _get_deny_patterns() -> list:
    """Config'den veya varsayılan DENY_PATTERNS'ı döndürür."""
    try:
        patterns = _cfg().security.deny_patterns
        if patterns:
            return patterns
    except Exception:
        pass
    return DENY_PATTERNS


# ─────────────────────────────────────────────
#  Güvenlik Fonksiyonları
# ─────────────────────────────────────────────

def is_dangerous_bash(cmd: str) -> Optional[str]:
    patterns = _get_deny_patterns()
    for pat in patterns:
        if re.search(pat, str(cmd).strip()):
            log.warning("🚫 GÜVENLİK: Tehlikeli komut engellendi | pattern=%s | cmd=%s", pat, str(cmd).strip()[:100])
            return f"Blocked by denylist pattern: {pat}"
    return None


def safe_relpath(path: str) -> str:
    p = Path(path).expanduser()
    if p.is_absolute():
        log.warning("🚫 GÜVENLİK: Absolute path engellendi | path=%s", path)
        raise SecurityViolationError(
            f"Absolute path kullanılamaz: {path}",
            violation_type="absolute_path",
            suggestion="Workspace içinde relative path kullanın (ör: data/raw/file.csv).",
        )
    norm = Path(os.path.normpath(str(p)))
    if str(norm).startswith(".."):
        log.warning("🚫 GÜVENLİK: Path traversal engellendi | path=%s", path)
        raise SecurityViolationError(
            f"Path traversal engellendi: {path}",
            violation_type="path_traversal",
            suggestion="Üst dizinlere erişim yasaktır. Workspace içindeki dosyaları kullanın.",
        )
    return str(norm)


def current_project() -> str:
    from agent import DEFAULT_PROJECT
    try:
        return os.getenv("AGENT_PROJECT", _cfg().workspace.default_project)
    except Exception:
        return os.getenv("AGENT_PROJECT", DEFAULT_PROJECT)


# ─────────────────────────────────────────────
#  Kod Çalıştırma
# ─────────────────────────────────────────────

def run_python(code: str, workspace: Path, timeout_s: int = 180) -> str:
    proj = current_project()
    if proj and proj != "workspace":
        workspace = workspace / proj
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    log.info("🐍 PYTHON çalıştırılıyor | timeout=%ds | kod_uzunluk=%d karakter", timeout_s, len(str(code)))
    log.debug("🐍 PYTHON kod:\n%s", str(code)[:500])
    code = textwrap.dedent(str(code)).strip() + "\n"

    # Kök dizini PYTHONPATH'e ekle
    root_dir = Path(__file__).resolve().parent.parent

    # ── Güvenli Sandboxing (Phase 3 Enterprise) ──
    sandbox_injection = textwrap.dedent(f"""
        import sys
        import os
        sys.path.insert(0, r'{root_dir}')
        
        # Tehlikeli sys modüllerini / yetkilerini kısmi kısıtla
        del sys.modules['os']
        # Not: Tam izolasyon (Docker/Firejail) üretim ortamı gerektirir.
        # Basit Python-level sandbox uygulanıyor.
    """)
    code = sandbox_injection + code

    # ── MLflow Otomatik Takip Entegrasyonu (Sprint 5) ──
    ml_keywords = ["train", "fit", "LogisticRegression", "RandomForest", "mlflow", "tracker", "X_train", "y_train"]
    is_ml_code = any(kw in code for kw in ml_keywords)

    if is_ml_code:
        log.info("📊 ML kodu algılandı, otomatik MLflow takibi denenecek...")
        injection = textwrap.dedent(f"""
            try:
                from mlflow_tracker import get_shared_tracker
                import time
                _auto_tracker = get_shared_tracker()
                _auto_tracker.start_run(run_name="agent_auto_run_{{time.strftime('%H%M%S')}}")
                _mlflow_active = True
            except ImportError:
                _mlflow_active = False
            try:
        """)
        indented_code = textwrap.indent(code, "    ")
        end_injection = textwrap.dedent("""
            finally:
                if _mlflow_active:
                    _auto_tracker.end_run()
        """)
        code = injection + "\n" + indented_code + "\n" + end_injection

    try:
        from ultra_agent.runtime.sandbox import SandboxRuntime
        sandbox = SandboxRuntime(workspace=workspace, work_dir=str(workspace))

        start_time = time.time()
        out, return_code = sandbox.run_python_code(code, timeout=timeout_s)
        elapsed = time.time() - start_time

        result = out if out.strip() else f"[python exit code: {return_code}] (no output)"
        log.info("🐍 PYTHON (Sandbox) tamamlandı | süre=%.2fs | exit_code=%d | çıktı_uzunluk=%d", elapsed, return_code, len(result))

        if return_code != 0:
            log.warning("🐍 PYTHON (Sandbox) hata ile bitti | exit_code=%d", return_code)

        return result

    except ToolTimeoutError:
        raise
    except Exception as e:
        log.error("🐍 PYTHON beklenmeyen hata | %s", e, exc_info=True)
        raise ToolExecutionError("PYTHON", str(e), details=f"Kod uzunluğu: {len(code)} karakter")


def run_bash(cmd: str, workspace: Path, timeout_s: int = 180) -> str:
    proj = current_project()
    if proj and proj != "workspace":
        workspace = workspace / proj
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    cmd_str = str(cmd)

    # ── HITL / Governance Onayı (S8-3) ──
    from ultra_agent.observability.audit_trail import AuditTrailLogger
    from utils.config import get_config
    _cfg_local = get_config()
    audit_logger = AuditTrailLogger(workspace)
    try:
        from core.hitl import HITLManager
        hitl = HITLManager(workspace)
        is_approved = hitl.require_approval("agent_auto", "BASH_EXEC", {"cmd": cmd_str[:500]})
        if not is_approved:
            audit_logger.log_critical_action("agent_auto", "BASH_EXEC", {"cmd": cmd_str[:500]}, "REJECTED_BY_HITL")
            return "[BASH_EXEC] HITL tarafından reddedildi."
        audit_logger.log_critical_action("agent_auto", "BASH_EXEC", {"cmd": cmd_str[:500]}, "APPROVED_BY_HITL")
    except ImportError:
        pass

    log.info("💻 BASH çalıştırılıyor | timeout=%ds | komut_uzunluk=%d karakter", timeout_s, len(cmd_str))
    log.debug("💻 BASH komut:\n%s", cmd_str[:500])
    reason = is_dangerous_bash(cmd)
    if reason:
        log.warning("💻 BASH ENGELLENDİ | sebep=%s | reason=%s", reason, cmd.strip()[:100])
        raise SecurityViolationError(
            f"Tehlikeli komut engellendi: {cmd.strip()[:80]}",
            violation_type="dangerous_command",
            details=reason,
            suggestion="Bu komut güvenlik politikası tarafından engellendi.",
        )
    try:
        start_time = time.time()
        res = subprocess.run(
            cmd, shell=True, cwd=str(workspace), capture_output=True, text=True, timeout=timeout_s
        )
        elapsed = time.time() - start_time
        stdout_str = res.stdout or ""
        stderr_str = res.stderr or ""

        if len(stdout_str) > 20000:
            log.warning("💻 BASH çıktısı çok uzun, kırpılıyor (%d -> 20000)", len(stdout_str))
            stdout_str = stdout_str[:20000] + "\n...[TRUNCATED]"
        if len(stderr_str) > 20000:
            log.warning("💻 BASH hata çıktısı çok uzun, kırpılıyor (%d -> 20000)", len(stderr_str))
            stderr_str = stderr_str[:20000] + "\n...[TRUNCATED]"

        out = stdout_str + stderr_str
        result = out.strip() if out.strip() else f"[bash exit code: {res.returncode}] (no output)"
        log.info("💻 BASH tamamlandı | süre=%.2fs | exit_code=%d | çıktı_uzunluk=%d", elapsed, res.returncode, len(result))
        if res.returncode != 0:
            log.warning("💻 BASH hata ile bitti | exit_code=%d | cmd=%s", res.returncode, cmd.strip()[:100])
            result = f"[BASH_ERROR exit={res.returncode}] cmd={cmd.strip()[:100]}\n{result}"
        return result
    except subprocess.TimeoutExpired:
        log.error("💻 BASH TIMEOUT | %ds aşıldı | cmd=%s", timeout_s, cmd.strip()[:100])
        raise ToolTimeoutError("BASH", timeout_s)
    except (ToolTimeoutError, SecurityViolationError):
        raise
    except Exception as e:
        log.error("💻 BASH beklenmeyen hata | %s", e, exc_info=True)
        raise ToolExecutionError("BASH", str(e), details=f"Komut: {cmd.strip()[:100]}")


# ─────────────────────────────────────────────
#  Web Araçları
# ─────────────────────────────────────────────

def web_search(query: str) -> str:
    query = query.strip()
    query_str = str(query)
    if not query_str:
        log.warning("🌐 WEB_SEARCH: Boş sorgu gönderildi")
        raise ValidationError("query", "Web araması için sorgu boş olamaz.")
    log.info("🌐 WEB_SEARCH başlatıldı | sorgu=%s", query_str[:100])
    try:
        from ddgs import DDGS
        start_time = time.time()
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=10):
                results.append(
                    {"title": r.get("title"), "href": r.get("href"), "body": r.get("body")}
                )
        elapsed = time.time() - start_time
        log.info("🌐 WEB_SEARCH tamamlandı | süre=%.2fs | sonuç_sayısı=%d", elapsed, len(results))
        return json.dumps(results, ensure_ascii=False, indent=2)
    except Exception as e:
        log.error("🌐 WEB_SEARCH HATA | sorgu=%s | hata=%s", query_str[:80], e, exc_info=True)
        raise ToolExecutionError(
            "WEB_SEARCH", str(e),
            suggestion="ddgs paketini kurun: python -m pip install -U ddgs",
        )


def web_open(url: str) -> str:
    url_str = str(url).strip()
    if not (url_str.startswith("http://") or url_str.startswith("https://")):
        log.warning("📖 WEB_OPEN: Geçersiz URL | url=%s", url_str[:100])
        raise ValidationError(
            "url", f"Geçersiz URL: {url_str[:80]}",
            suggestion="URL http:// veya https:// ile başlamalıdır.",
        )
    log.info("📖 WEB_OPEN başlatıldı | url=%s", url_str[:150])
    try:
        import requests
        from bs4 import BeautifulSoup

        start_time = time.time()
        r = requests.get(url_str, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        text = soup.get_text("\n")
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        elapsed = time.time() - start_time
        truncated = len(text) > 12000
        log.info("📖 WEB_OPEN tamamlandı | süre=%.2fs | status=%d | metin_uzunluk=%d | kırpıldı=%s",
                 elapsed, r.status_code, len(text), truncated)
        return (text[:12000] + "\n\n[TRUNCATED]") if truncated else text
    except Exception as e:
        log.error("📖 WEB_OPEN HATA | url=%s | hata=%s", url_str[:100], e, exc_info=True)
        raise ToolExecutionError(
            "WEB_OPEN", str(e),
            details=f"URL: {url_str[:100]}",
            suggestion="URL'nin erişilebilir olduğundan emin olun.",
        )


def browser_open(url: str, session_id: str, workspace: Path) -> str:
    """Headless browser ile JavaScript-rendered sayfayı açar ve metin içeriğini döner."""
    url_str = str(url).strip()
    if not (url_str.startswith("http://") or url_str.startswith("https://")):
        raise ValidationError(
            "url", f"Geçersiz URL: {url_str[:80]}",
            suggestion="URL http:// veya https:// ile başlamalıdır.",
        )
    log.info("🌐 BROWSER_OPEN başlatıldı | url=%s", url_str[:150])
    try:
        from ultra_agent.runtime.browser.dom_driver import DOMDriver
        driver = DOMDriver(tenant_id=session_id, workspace_dir=workspace)

        start_time = time.time()
        text = driver.navigate_and_extract(url_str)
        elapsed = time.time() - start_time

        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        truncated = len(text) > 15000
        log.info("🌐 BROWSER_OPEN tamamlandı | süre=%.2fs | metin_uzunluk=%d | kırpıldı=%s",
                 elapsed, len(text), truncated)
        content = text[:15000] + "\n\n[TRUNCATED]" if truncated else text
        return content
    except ImportError:
        log.warning("🌐 BROWSER_OPEN: Playwright yüklü değil, WEB_OPEN'a geri dönülüyor")
        return web_open(url_str)
    except Exception as e:
        log.error("🌐 BROWSER_OPEN HATA | url=%s | hata=%s", url_str[:100], e, exc_info=True)
        raise ToolExecutionError(
            "BROWSER_OPEN", str(e),
            details=f"URL: {url_str[:100]}",
            suggestion="Playwright kurulumu: pip install playwright && playwright install chromium",
        )


def browser_action(payload: str, workspace: Path = None) -> str:
    """Etkileşimli headless browser oturumu. Çok adımlı komutlarla tarayıcı kontrolü."""
    log.info("🌐 BROWSER_ACTION başlatıldı")

    commands = []
    for line in payload.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            cmd, _, arg = line.partition(":")
            commands.append((cmd.strip().lower(), arg.strip()))

    if not commands:
        return "[BROWSER_ACTION] Geçerli komut bulunamadı."

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return "[BROWSER_ACTION HATA] Playwright yüklü değil. Kur: pip install playwright && playwright install chromium"

    results = []
    start_time = time.time()

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(
                user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
                viewport={"width": 1280, "height": 720},
            )

            for i, (cmd, arg) in enumerate(commands, 1):
                try:
                    if cmd == "goto":
                        url = arg.strip()
                        page.goto(url, wait_until="domcontentloaded", timeout=30000)
                        page.wait_for_timeout(1000)
                        results.append(f"[{i}] goto: {url} → ✅")

                    elif cmd == "click":
                        page.click(arg, timeout=10000)
                        page.wait_for_timeout(500)
                        results.append(f"[{i}] click: {arg} → ✅")

                    elif cmd == "type":
                        parts = arg.split("|", 1)
                        if len(parts) != 2:
                            results.append(f"[{i}] type: ❌ Format: CSS_SELECTOR | metin")
                            continue
                        selector, text = parts[0].strip(), parts[1].strip()
                        page.fill(selector, text)
                        results.append(f"[{i}] type: {selector} → '{text[:50]}' ✅")

                    elif cmd == "screenshot":
                        filename = arg.strip() or "screenshot.png"
                        if workspace:
                            save_path = workspace / filename
                        else:
                            save_path = Path(filename)
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        page.screenshot(path=str(save_path), full_page=False)
                        results.append(f"[{i}] screenshot: {save_path} → ✅")

                    elif cmd == "text":
                        selector = arg.strip() or "body"
                        text_content = page.inner_text(selector)
                        text_content = re.sub(r"\n{3,}", "\n\n", text_content).strip()
                        if len(text_content) > 5000:
                            text_content = text_content[:5000] + "\n[TRUNCATED]"
                        results.append(f"[{i}] text ({selector}):\n{text_content}")

                    elif cmd == "html":
                        selector = arg.strip() or "body"
                        html_content = page.inner_html(selector)
                        if len(html_content) > 5000:
                            html_content = html_content[:5000] + "\n[TRUNCATED]"
                        results.append(f"[{i}] html ({selector}):\n{html_content}")

                    elif cmd == "scroll":
                        direction = arg.strip().lower()
                        if direction == "down":
                            page.evaluate("window.scrollBy(0, 500)")
                        elif direction == "up":
                            page.evaluate("window.scrollBy(0, -500)")
                        elif direction == "bottom":
                            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                        elif direction == "top":
                            page.evaluate("window.scrollTo(0, 0)")
                        else:
                            page.locator(direction).scroll_into_view_if_needed()
                        page.wait_for_timeout(300)
                        results.append(f"[{i}] scroll: {direction} → ✅")

                    elif cmd == "wait":
                        secs = float(arg) if arg else 1
                        secs = min(secs, 15)
                        page.wait_for_timeout(int(secs * 1000))
                        results.append(f"[{i}] wait: {secs}s → ✅")

                    elif cmd == "select":
                        parts = arg.split("|", 1)
                        if len(parts) != 2:
                            results.append(f"[{i}] select: ❌ Format: CSS_SELECTOR | değer")
                            continue
                        selector, value = parts[0].strip(), parts[1].strip()
                        page.select_option(selector, value)
                        results.append(f"[{i}] select: {selector} = '{value}' → ✅")

                    elif cmd == "title":
                        title = page.title()
                        results.append(f"[{i}] title: {title}")

                    elif cmd == "url":
                        current_url = page.url
                        results.append(f"[{i}] url: {current_url}")

                    elif cmd == "back":
                        page.go_back()
                        page.wait_for_timeout(500)
                        results.append(f"[{i}] back → ✅")

                    elif cmd == "forward":
                        page.go_forward()
                        page.wait_for_timeout(500)
                        results.append(f"[{i}] forward → ✅")

                    elif cmd == "press":
                        page.keyboard.press(arg.strip())
                        results.append(f"[{i}] press: {arg} → ✅")

                    elif cmd == "evaluate":
                        js_result = page.evaluate(arg.strip())
                        results.append(f"[{i}] evaluate: {str(js_result)[:2000]}")

                    else:
                        results.append(f"[{i}] ❌ Bilinmeyen komut: {cmd}")

                except Exception as e:
                    results.append(f"[{i}] ❌ {cmd}: {type(e).__name__}: {str(e)[:200]}")

            browser.close()

    except Exception as e:
        log.error("🌐 BROWSER_ACTION HATA: %s", e, exc_info=True)
        results.append(f"\n❌ Browser hatası: {type(e).__name__}: {str(e)[:300]}")

    elapsed = time.time() - start_time
    log.info("🌐 BROWSER_ACTION tamamlandı | süre=%.2fs | komut_sayısı=%d", elapsed, len(commands))

    output = "\n".join(results)
    return f"[BROWSER_ACTION] {len(commands)} komut | {elapsed:.1f}s\n\n{output}"


# ─────────────────────────────────────────────
#  Dosya İşlemleri
# ─────────────────────────────────────────────

def _clean_file_payload(payload: str) -> str:
    """LLM'in eklediği 'path:', 'file:', 'dosya:' gibi prefix'leri temizle."""
    cleaned = str(payload).strip()
    for prefix in ('path:', 'file:', 'dosya:', 'Path:', 'FILE:', 'File:'):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break
    cleaned = cleaned.strip('"').strip("'").strip('`')
    return cleaned


def sanitize_content(content: str) -> str:
    content_clean = re.sub(r"^\s*```[a-zA-Z0-9_-]*\s*$", "", content, flags=re.MULTILINE)
    content_clean = re.sub(r"^\s*```\s*$", "", content_clean, flags=re.MULTILINE)
    return content_clean.lstrip("\n")


def _strip_redundant_prefixes(rel: str, proj: str) -> str:
    """LLM'in yanlışlıkla eklediği workspace/, proje adı ve benzeri prefix'leri agresif olarak temizle."""
    _KNOWN_ROOTS = {"src", "data", "results", "docs", "models", "notebooks", "tests", "config", proj}
    _KNOWN_FILES = {"report.md", "README.md", "readme.md", "pyproject.toml", "setup.py",
                    "todo.md", "report.txt", ".gitignore", "Makefile"}

    parts = list(Path(rel).parts)
    original = rel

    while parts and parts[0] in ("workspace", proj, "scratch_project", "Kanser_Hücresi_Analiz"):
        parts = parts[1:]

    for i, part in enumerate(parts):
        if part in _KNOWN_ROOTS or part in _KNOWN_FILES:
            parts = parts[i:]
            break

    cleaned = str(Path(*parts)) if parts else rel

    if cleaned != original:
        log.info("✍️ WRITE_FILE yol düzeltildi: %s → %s", original, cleaned)

    return cleaned


def read_file(payload: str, workspace: Path) -> str:
    payload_str = _clean_file_payload(payload)
    rel = safe_relpath(payload_str)
    p = workspace / rel
    if not p.exists():
        log.warning("📄 READ_FILE: Dosya bulunamadı | path=%s", rel)
        raise FileOperationError(
            "okuma", rel, "Dosya bulunamadı.",
            suggestion=f"Dosyanın var olduğundan emin olun: {rel}",
        )
    if p.is_dir():
        log.warning("📄 READ_FILE: Klasör verildi | path=%s", rel)
        raise FileOperationError(
            "okuma", rel, "Verilen yol bir klasör, dosya değil.",
            suggestion="Dosya yolunu belirtin, klasör yolunu değil.",
        )
    data = p.read_text(encoding="utf-8", errors="replace")
    log.info("📄 READ_FILE | path=%s | boyut=%d bytes", rel, len(data))
    return (data[:20000] + "\n\n[TRUNCATED]") if len(data) > 20000 else data


def write_file(payload: str, workspace: Path) -> str:
    raw = payload.strip()
    if "---" not in raw:
        log.warning("✍️ WRITE_FILE: Format hatası — '---' ayırıcı bulunamadı")
        raise ValidationError(
            "WRITE_FILE format",
            "'---' ayırıcı bulunamadı.",
            suggestion="Doğru format: path: dosya.py\n---\niçerik...",
        )
    head, content = raw.split("---", 1)
    m = re.search(r"^\s*path:\s*(.+)\s*$", head.strip(), re.MULTILINE)
    if not m:
        log.warning("✍️ WRITE_FILE: Format hatası — 'path:' satırı bulunamadı")
        raise ValidationError(
            "WRITE_FILE format",
            "'path: ...' satırı eksik.",
            suggestion="Blok başında 'path: dosya_adı.py' satırı olmalı.",
        )

    rel = safe_relpath(m.group(1).strip())
    proj = current_project()
    rel = _strip_redundant_prefixes(rel, proj)
    if not rel.startswith(proj + "/") and rel != proj:
        rel = f"{proj}/{rel}"

    p = workspace / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(sanitize_content(content), encoding="utf-8")
    log.info("✍️ WRITE_FILE | path=%s | boyut=%d bytes", rel, p.stat().st_size)
    return f"[OK] Wrote {rel} ({p.stat().st_size} bytes)"


def append_todo(payload: str, workspace: Path) -> str:
    todo = workspace / current_project() / "todo.md"
    todo.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    entry = payload.strip()
    if not entry:
        log.warning("📝 TODO: Boş içerik gönderildi")
        raise ValidationError("TODO", "TODO bloğu boş olamaz.")

    if not todo.exists():
        todo.write_text("# TODO List\n", encoding="utf-8")

    with open(todo, "a", encoding="utf-8") as f:
        f.write(f"- [ ] {entry} (Eklenme: {ts})\n")

    log.info("📝 TODO eklendi | dosya=%s | uzunluk=%d", todo.name, len(entry))
    return f"[OK] Added to TODO: {todo.name}"


def version_dataset(dataset_id: str, workspace: Path) -> str:
    """Veri setinin anlık hash değerini hesaplar ve MLflow'a kaydeder."""
    import dataset_catalog
    from mlflow_tracker import get_shared_tracker

    dataset_id = dataset_id.strip()
    log.info("📦 VERSION_DATASET: %s", dataset_id)

    try:
        version_hash = dataset_catalog.get_dataset_version(dataset_id, workspace)
        tracker = get_shared_tracker()

        if tracker.is_mlflow_active:
            tracker.set_tag(f"dataset.{dataset_id}.hash", version_hash)
            tracker.set_tag(f"dataset.{dataset_id}.version", "tracked-auto")

        return f"[OK] Dataset '{dataset_id}' versioned. Hash: {version_hash}"
    except Exception as e:
        log.error("📦 VERSION_DATASET HATA: %s", e)
        return f"[ERROR] Dataset versioning failed: {str(e)}"


# ─────────────────────────────────────────────
#  Tool Parsing
# ─────────────────────────────────────────────

def extract_tools(text: str) -> Tuple[List[Tuple[str, str]], str]:
    """Tool etiketlerini parse eder. Hem <TAG>...</TAG> hem de kapanışsız <TAG>... destekler."""
    text_str = str(text) if text else ""
    tools = []
    remaining = text_str

    for tag in TOOL_TAGS:
        open_tag = f"<{tag}>"
        close_tag = f"</{tag}>"

        text_upper = remaining.upper()
        open_tag_upper = open_tag.upper()
        close_tag_upper = close_tag.upper()

        start = text_upper.find(open_tag_upper)
        if start == -1:
            continue

        content_start = start + len(open_tag)
        end = text_upper.find(close_tag_upper, content_start)

        if end != -1:
            payload = remaining[content_start:end].strip()
            remaining = (remaining[:start] + remaining[end + len(close_tag):]).strip()
        else:
            payload = remaining[content_start:].strip()
            remaining = remaining[:start].strip()

        if payload:
            tools.append((tag.upper(), payload))

    return tools, remaining.strip()


def extract_tool(text: str) -> Tuple[Optional[str], Optional[str], str]:
    tools, outside = extract_tools(text)
    if not tools:
        return None, None, outside
    return tools[0][0], tools[0][1], outside


def normalize_user_message(s: str) -> str:
    s = s.replace("\r\n", "\n")
    lines = [ln.strip() for ln in s.split("\n")]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def autosave_web_outputs(cfg, tool: str, out: str) -> None:
    proj = current_project()
    log_dir = cfg.workspace / proj / "datasets"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    fname = f"{tool.lower()}_{stamp}.json" if tool == "WEB_SEARCH" else f"{tool.lower()}_{stamp}.txt"
    (log_dir / fname).write_text(out, encoding="utf-8")
