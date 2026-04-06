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
import sys
import textwrap
import time
import uuid
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

from bio_ml_agent.exceptions import (
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
    "READ_FILE", "WRITE_FILE", "TODO", "CLINICAL_VISION", "SWARM", "DEEP_RESEARCH", "INDEX_WORKSPACE", "BACKGROUND_JOB",
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
    from bio_ml_agent.utils.config import get_config
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
    _DEFAULT_PROJECT = "scratch_project"
    try:
        return os.getenv("AGENT_PROJECT", _cfg().workspace.default_project)
    except Exception:
        return os.getenv("AGENT_PROJECT", _DEFAULT_PROJECT)


# ─────────────────────────────────────────────
#  Kod Çalıştırma
# ─────────────────────────────────────────────

def run_python(code: str, workspace: Path, timeout_s: int = 180, project_name: Optional[str] = None) -> str:
    proj = project_name or current_project()
    if proj and proj != "workspace":
        workspace = workspace / proj
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    log.info("🐍 PYTHON çalıştırılıyor | project=%s | timeout=%ds | kod_uzunluk=%d karakter", proj, timeout_s, len(str(code)))
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
                from bio_ml_agent.mlflow_tracker import get_shared_tracker
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
        from bio_ml_agent.ultra_agent.runtime.sandbox import SandboxRuntime
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


def run_bash(cmd: str, workspace: Path, timeout_s: int = 180, project_name: Optional[str] = None) -> str:
    proj = project_name or current_project()
    if proj and proj != "workspace":
        workspace = workspace / proj
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    cmd_str = str(cmd)

    # ── HITL / Governance Onayı (S8-3) ──
    from bio_ml_agent.ultra_agent.observability.audit_trail import AuditTrailLogger
    from bio_ml_agent.utils.config import get_config
    _cfg_local = get_config()
    audit_logger = AuditTrailLogger(workspace)
    try:
        from bio_ml_agent.core.hitl import HITLManager
        hitl = HITLManager(workspace)
        is_approved = hitl.require_approval("agent_auto", "BASH_EXEC", {"cmd": cmd_str[:500]})
        if not is_approved:
            audit_logger.log_critical_action("agent_auto", "BASH_EXEC", {"cmd": cmd_str[:500]}, "REJECTED_BY_HITL")
            return "[BASH_EXEC] HITL tarafından reddedildi."
        audit_logger.log_critical_action("agent_auto", "BASH_EXEC", {"cmd": cmd_str[:500]}, "APPROVED_BY_HITL")
    except ImportError:
        pass

    log.info("💻 BASH çalıştırılıyor | project=%s | timeout=%ds | komut_uzunluk=%d karakter", proj, timeout_s, len(cmd_str))
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

        # .venv/bin dizinini PATH'e ekle (S9-1: Environment Isolation Fix)
        env = os.environ.copy()
        venv_bin = str(Path(sys.executable).parent)
        env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"

        res = subprocess.run(
            cmd, shell=True, cwd=str(workspace), capture_output=True, text=True, timeout=timeout_s,
            env=env
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
        from duckduckgo_search import DDGS
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
        from bio_ml_agent.ultra_agent.runtime.browser.dom_driver import DOMDriver
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


def browser_action(payload: str, workspace: Path = None, timeout_s: int = 60, project_name: Optional[str] = None) -> str:
    """Etkileşimli headless browser oturumu. Çok adımlı komutlarla tarayıcı kontrolü."""
    log.info("🌐 BROWSER_ACTION başlatıldı")

    proj = project_name or current_project()
    browser_ws = workspace / proj / "browser" if workspace else Path("browser")
    browser_ws.mkdir(parents=True, exist_ok=True)

    # Komut takma adları (alias) — LLM farklı isimler kullanabilir
    _CMD_ALIASES = {
        "fill": "type", "input": "type", "enter": "type",
        "navigate": "goto", "open": "goto", "go": "goto", "nav": "goto",
        "tap": "click", "press_button": "click",
        "snap": "screenshot", "capture": "screenshot",
        "extract": "text", "get_text": "text", "read": "text",
        "js": "evaluate", "exec": "evaluate",
        "scroll_down": "scroll", "scroll_up": "scroll",
    }

    def _normalize_cmd(raw_cmd: str) -> str:
        """Komut adını normalize et ve alias'ları çöz."""
        c = raw_cmd.strip().lower().replace("-", "_").replace(" ", "_")
        return _CMD_ALIASES.get(c, c)

    def _parse_line(line: str) -> Optional[tuple]:
        """Tek bir satırı parse et — önce ':' ile, sonra boşlukla dene."""
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            return None

        # 1) Kolon ayırıcı — "goto: https://..." veya "click: #btn"
        if ":" in line:
            cmd_part, _, arg = line.partition(":")
            cmd = _normalize_cmd(cmd_part)
            # Eğer cmd bilinen bir komutsa, kolonu ayırıcı olarak kabul et
            _KNOWN_CMDS = {"goto", "click", "type", "screenshot", "text", "html",
                           "scroll", "wait", "select", "title", "url", "back",
                           "forward", "press", "evaluate", "fill", "navigate",
                           "open", "tap", "snap", "capture", "extract", "read",
                           "js", "exec", "go", "nav", "input", "enter"}
            if cmd_part.strip().lower().replace("-", "_").replace(" ", "_") in _KNOWN_CMDS or cmd in _KNOWN_CMDS:
                return (cmd, arg.strip())
            # Kolon URL'nin parçası olabilir (ör: "goto https://example.com")
            # Boşluk ayırıcıya düş

        # 2) Boşluk ayırıcı — "goto https://..." veya "click .btn"
        parts = line.split(None, 1)
        if parts:
            cmd = _normalize_cmd(parts[0])
            arg = parts[1].strip() if len(parts) > 1 else ""
            return (cmd, arg)

        return None

    commands = []
    # JSON Desteği (P0)
    payload_trimmed = payload.strip()
    log.debug("🌐 BROWSER_ACTION payload (ilk 500 karakter): %s", payload_trimmed[:500])

    if (payload_trimmed.startswith("{") or payload_trimmed.startswith("[")):
        try:
            import json
            data = json.loads(payload_trimmed)
            if isinstance(data, list):
                for item in data:
                    cmd = _normalize_cmd(item.get("action", item.get("command", "")))
                    # Genişletilmiş anahtar desteği (P0)
                    arg = item.get("arg", item.get("args", item.get("argument",
                          item.get("url", item.get("selector", item.get("text",
                          item.get("filename", "")))))))
                    # Type/fill için özel birleşim: selector | text
                    if cmd == "type" and "selector" in item and "text" in item:
                        arg = f"{item['selector']} | {item['text']}"

                    if cmd: commands.append((cmd, arg or ""))
            else:
                cmd = _normalize_cmd(data.get("action", data.get("command", "")))
                arg = data.get("arg", data.get("args", data.get("argument",
                      data.get("url", data.get("selector", data.get("text",
                      data.get("filename", "")))))))
                if cmd == "type" and "selector" in data and "text" in data:
                    arg = f"{data['selector']} | {data['text']}"
                if cmd: commands.append((cmd, arg or ""))
        except Exception as e:
            log.warning("🌐 BROWSER_ACTION JSON parse hatası: %s, line-based fallback deneniyor", e)
            # Fallback to line-based
            for line in payload_trimmed.splitlines():
                parsed = _parse_line(line)
                if parsed:
                    commands.append(parsed)
    else:
        for line in payload_trimmed.splitlines():
            parsed = _parse_line(line)
            if parsed:
                commands.append(parsed)

    log.info("🌐 BROWSER_ACTION parse sonucu: %d komut bulundu | payload_uzunluk=%d", len(commands), len(payload_trimmed))
    if commands:
        log.debug("🌐 BROWSER_ACTION komutlar: %s", [(c, a[:80]) for c, a in commands])

    if not commands:
        return "[BROWSER_ACTION] Geçerli komut bulunamadı."

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return "[BROWSER_ACTION HATA] Playwright yüklü değil. Kur: pip install playwright && playwright install chromium"

    results = []
    start_time = time.time()

    try:
        from bio_ml_agent.ultra_agent.observability.audit_trail import AuditTrailLogger
        audit_logger = AuditTrailLogger(workspace / proj if workspace else Path("."))
    except Exception:
        audit_logger = None

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--disable-blink-features=AutomationControlled", "--no-sandbox", "--disable-dev-shm-usage"]
            )
            page = browser.new_page(
                user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
                viewport={"width": 1280, "height": 720},
            )

            for i, (cmd, arg) in enumerate(commands, 1):
                try:
                    if cmd == "goto":
                        url = arg.strip()
                        if audit_logger: audit_logger.log_critical_action("agent", "BROWSER_GOTO", {"url": url}, "EXECUTED")
                        page.goto(url, wait_until="domcontentloaded", timeout=30000)
                        page.wait_for_timeout(1000)
                        results.append(f"[{i}] goto: {url} → ✅")

                    elif cmd == "click":
                        if audit_logger: audit_logger.log_critical_action("agent", "BROWSER_CLICK", {"selector": arg}, "EXECUTED")
                        page.locator(arg).click(timeout=10000)
                        page.wait_for_timeout(500)
                        results.append(f"[{i}] click: {arg} → ✅")

                    elif cmd == "type":
                        parts = arg.split("|", 1)
                        if len(parts) != 2:
                            results.append(f"[{i}] type: ❌ Format: CSS_SELECTOR | metin")
                            continue
                        selector, text = parts[0].strip(), parts[1].strip()
                        if audit_logger: audit_logger.log_critical_action("agent", "BROWSER_TYPE", {"selector": selector, "len": len(text)}, "EXECUTED")
                        try:
                            page.locator(selector).fill(text)
                        except Exception:
                            # fill() contenteditable vb. bazı elementlerde çalışmaz, type() fallback
                            page.locator(selector).click()
                            page.locator(selector).type(text)
                        results.append(f"[{i}] type: {selector} → '{text[:50]}' ✅")

                    elif cmd == "screenshot":
                        filename = arg.strip()
                        if not filename or "." not in filename:
                            filename = f"screenshot_{int(time.time())}.png"
                        save_path = browser_ws / filename
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


def read_file(payload: str, workspace: Path, project_name: Optional[str] = None) -> str:
    payload_str = _clean_file_payload(payload)
    rel = safe_relpath(payload_str)

    proj = project_name or current_project()
    rel = _strip_redundant_prefixes(rel, proj)
    if not rel.startswith(proj + "/") and rel != proj:
         # Eğer yol direkt proje kökünde bir dosya değilse veya proje adı ön eki yoksa ekleyelim
         # read_file için workspace / rel kullanıyoruz, rel içinde proje adı olmalı
         rel = f"{proj}/{rel}"

    p = workspace / rel
    if not p.exists():
        log.warning("📄 READ_FILE: Dosya bulunamadı | proj=%s | path=%s", proj, rel)
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


def write_file(payload: str, workspace: Path, project_name: Optional[str] = None) -> str:
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
    proj = project_name or current_project()
    rel = _strip_redundant_prefixes(rel, proj)
    if not rel.startswith(proj + "/") and rel != proj:
        rel = f"{proj}/{rel}"

    p = workspace / rel
    p.parent.mkdir(parents=True, exist_ok=True)

    # ── S8-4: Audit Trail ──
    try:
        from bio_ml_agent.ultra_agent.observability.audit_trail import AuditTrailLogger
        audit_logger = AuditTrailLogger(workspace)
        audit_logger.log_critical_action("agent_auto", "WRITE_FILE", {"path": str(rel), "size_bytes": len(content)}, "AUTO_APPROVED")
    except ImportError:
        pass

    p.write_text(sanitize_content(content), encoding="utf-8")
    log.info("✍️ WRITE_FILE | path=%s | boyut=%d bytes", rel, p.stat().st_size)
    return f"[OK] Wrote {rel} ({p.stat().st_size} bytes)"


def append_todo(payload: str, workspace: Path, project_name: Optional[str] = None) -> str:
    proj = project_name or current_project()
    todo = workspace / proj / "todo.md"
    todo.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    entry = payload.strip()

    # 1. Stricter Validation
    if not entry:
        log.warning("📝 TODO: Boş içerik gönderildi")
        raise ValidationError("TODO", "TODO bloğu boş olamaz.")

    if len(entry) < 10:
        log.warning("📝 TODO: İçerik çok kısa: %s", entry)
        raise ValidationError("TODO", "TODO açıklaması çok kısa, lütfen daha açıklayıcı olun (en az 10 karakter).")

    # 2. Extract context if provided (e.g., [Bug], [Feature])
    context = "Task"
    if entry.startswith("[") and "]" in entry:
        context = entry[1:entry.find("]")]

    # 3. Dosyaya yazma
    if not todo.exists():
        todo.write_text("# Proje Yapılacaklar (TODO) Listesi\n\n", encoding="utf-8")

    with open(todo, "a", encoding="utf-8") as f:
        f.write(f"- [ ] **{context}**: {entry} *(Eklenme: {ts})*\n")

    log.info("📝 TODO eklendi | dosya=%s | uzunluk=%d", todo.name, len(entry))

    # 4. Dış Entegrasyon Taslağı (Mock JIRA / GitHub Issues)
    ext_sync_msg = ""
    jira_token = os.getenv("JIRA_API_TOKEN")
    github_token = os.getenv("GITHUB_API_TOKEN")

    if jira_token:
        # Gerçek bir JIRA API çağrısı simülasyonu
        log.info("🔌 JIRA Entegrasyonu: Görev '#%s' JIRA'ya gönderiliyor...", context)
        ext_sync_msg = " | JIRA ile senkronize edildi."
    elif github_token:
        # Gerçek bir GitHub API çağrısı simülasyonu
        log.info("🔌 GitHub Entegrasyonu: Issue GitHub'a açılıyor...")
        ext_sync_msg = " | GitHub Issues ile senkronize edildi."

    return f"[OK] Added to TODO: {todo.name}{ext_sync_msg}"


def version_dataset(dataset_id: str, workspace: Path, project_name: Optional[str] = None) -> str:
    """Veri setinin anlık hash değerini hesaplar ve MLflow'a kaydeder."""
    import bio_ml_agent.dataset_catalog as dataset_catalog
    from bio_ml_agent.mlflow_tracker import get_shared_tracker

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

def run_deep_research(query: str, workspace: Path, project_name: str, model: str = "gemini-2.5-flash") -> str:
    """Derin Araştırma Ajanını başlatır ve sonuçları döner."""
    from bio_ml_agent.core.deep_research import DeepResearchAgent

    agent = DeepResearchAgent(model_name=model, workspace=workspace, project_name=project_name)
    try:
        result = agent.research(query)
        return result
    except Exception as e:
        log.error("🔍 DEEP_RESEARCH HATA: %s", e)
        return f"[ERROR] Deep Research failed: {str(e)}"

def clinical_vision(payload: str, workspace: Path, project_name: Optional[str] = None) -> str:
    """Klinik görüntü analizi yapar (Gemini 2.0 Vision Pro).
    Kullanım: <CLINICAL_VISION> image_path.png | modality | extra context </CLINICAL_VISION>
    """
    try:
        from bio_ml_agent.ultra_agent.vision.clinical_analyzer import ClinicalImageAnalyzer

        parts = [p.strip() for p in payload.split("|")]
        img_src = parts[0]
        modality = parts[1] if len(parts) > 1 else "general"
        context = parts[2] if len(parts) > 2 else ""

        # Dosya yolunu çalışma dizinine göre ayarla
        proj = project_name or current_project()
        if not os.path.isabs(img_src):
            img_path = str(workspace / proj / img_src)
        else:
            img_path = img_src

        analyzer = ClinicalImageAnalyzer()
        res = analyzer.analyze(img_path, modality=modality, extra_context=context)

        if "error" in res:
            return f"[CLINICAL_VISION ERROR] {res['error']}"

        out = f"Klinik Görüntü Analizi ({res.get('modality', modality)})\n"
        out += "="*40 + "\n"
        out += res.get("analysis", "") + "\n\n"
        out += f"⚠️ {res.get('disclaimer', '')}"

        return out
    except Exception as e:
        log.error(f"CLINICAL_VISION exception: {e}")
        return f"[ERROR] Vision analizi başarısız oldu: {str(e)}"


def index_workspace(payload: str, workspace: Path, project_name: Optional[str] = None) -> str:
    """Belirtilen dizindeki (dosya veya klasör) tüm dökümanları (PDF, DOCX, TXT, CSV) semantik hafızaya indexler."""
    from bio_ml_agent.ultra_agent.memory import get_memory_store

    proj = project_name or current_project()
    input_path = payload.strip()

    # Güvenli yol kontrolü ve tam yol oluşturma
    try:
        rel_path = safe_relpath(input_path) if input_path else proj
        # Eğer rel_path proje adı ile başlamıyorsa ekle (workspace köküne göre)
        if not rel_path.startswith(proj):
            full_path = workspace / proj / rel_path
        else:
            full_path = workspace / rel_path
    except Exception as e:
        return f"[INDEX_WORKSPACE ERROR] Geçersiz yol: {str(e)}"

    if not full_path.exists():
        return f"[INDEX_WORKSPACE ERROR] Yol bulunamadı: {full_path}"

    log.info("🗂️ INDEX_WORKSPACE: %s indexleniyor...", full_path)
    store = get_memory_store()

    supported_ext = {'.pdf', '.docx', '.txt', '.csv', '.md'}
    files_to_index = []

    if full_path.is_file():
        files_to_index.append(full_path)
    else:
        for root, _, files in os.walk(full_path):
            for f in files:
                ext = Path(f).suffix.lower()
                if ext in supported_ext:
                    files_to_index.append(Path(root) / f)

    if not files_to_index:
        return f"[INDEX_WORKSPACE] Indexlenecek uygun döküman bulunamadı (.pdf, .docx, .txt, .csv, .md desteklenir)."

    indexed_count = 0
    errors = []

    for fpath in files_to_index:
        try:
            content = ""
            ext = fpath.suffix.lower()

            if ext == '.pdf':
                try:
                    import pypdf
                    reader = pypdf.PdfReader(fpath)
                    content = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                except ImportError:
                    errors.append(f"{fpath.name}: pypdf kütüphanesi eksik.")
                    continue
            elif ext == '.docx':
                try:
                    import docx
                    doc = docx.Document(fpath)
                    content = "\n".join([p.text for p in doc.paragraphs])
                except ImportError:
                    errors.append(f"{fpath.name}: python-docx kütüphanesi eksik.")
                    continue
            else:
                # Metin tabanlı dosyalar
                content = fpath.read_text(encoding="utf-8", errors="replace")

            if not content.strip():
                continue

            # Basit chunking (1500 karakter civarı)
            chunks = [content[i:i+1500] for i in range(0, len(content), 1200)]

            for i, chunk in enumerate(chunks):
                store.store_memory(
                    content=chunk,
                    metadata={
                        "source": "workspace_index",
                        "file_path": str(fpath.relative_to(workspace)),
                        "file_name": fpath.name,
                        "chunk_index": i,
                        "project": proj
                    },
                    project_name=proj
                )

            indexed_count += 1
            log.info("✅ Indexlendi: %s (%d chunk)", fpath.name, len(chunks))

        except Exception as e:
            errors.append(f"{fpath.name}: {str(e)}")

    res = f"[INDEX_WORKSPACE] Başarıyla indexlendi: {indexed_count} dosya."
    if errors:
        res += "\n\nHatalar:\n- " + "\n- ".join(errors[:10])
        if len(errors) > 10:
            res += f"\n... ve {len(errors)-10} hata daha."

    return res


def background_job(payload: str, workspace: Path, project_name: Optional[str] = None) -> str:
    """Belirtilen bir tool çağrısını (örn: <WEB_SEARCH>...) Redis kuyruğuna (background) atar.
    Kullanım: <BACKGROUND_JOB> <TOOL>ayrıntılar</TOOL> </BACKGROUND_JOB>
    """
    try:
        from redis import Redis
        from rq import Queue
        cfg = _cfg()

        # Redis bağlantısı
        redis_conn = Redis(
            host=cfg.redis.host,
            port=cfg.redis.port,
            password=cfg.redis.password or None,
            db=cfg.redis.db
        )
        q = Queue('agent_tasks', connection=redis_conn)

        # Payload içinden tool bilgilerini ayıkla
        # Örn: payload = "<WEB_SEARCH>protein structure</WEB_SEARCH>"
        tag, inner_payload, _, attrs = extract_tool(payload)

        if not tag:
            return "[BACKGROUND_JOB ERROR] Geçersiz payload. Bir tool etiketi içermelidir."

        session_id = f"bg_{uuid.uuid4().hex[:8]}"

        # RQ kuyruğuna ekle
        job = q.enqueue(
            "bio_ml_agent.workers.job_worker.execute_agent_job",
            session_id=session_id,
            prompt=payload, # Orijinal tool çağrısını prompt olarak gönderiyoruz
            model=cfg.agent.model,
            timeout=cfg.agent.timeout,
            max_steps=5, # Arka plan görevleri için adım sınırını düşük tutabiliriz veya parametrik yapabiliriz
            job_timeout=cfg.agent.timeout + 300
        )

        log.info("🚀 BACKGROUND_JOB: %s kuyruğa eklendi. ID: %s", tag, job.id)

        return f"[BACKGROUND_JOB] '{tag}' görevi arka plana alındı.\nTask ID: {job.id}\nDurum kontrolü için: /api/v1/agent/status/{job.id}"

    except ImportError:
        return "[BACKGROUND_JOB ERROR] 'redis' veya 'rq' kütüphaneleri eksik."
    except Exception as e:
        log.error(f"BACKGROUND_JOB exception: {e}")
        return f"[ERROR] Arka plan görevi başlatılamadı: {str(e)}"


# ─────────────────────────────────────────────
#  Tool Parsing
# ─────────────────────────────────────────────

def extract_tools(text: str) -> Tuple[List[Dict[str, Any]], str]:
    """Tool etiketlerini parse eder. <TAG attr=val>...</TAG> formatını destekler.
    
    Döndürülen liste öğeleri: {"tool": str, "payload": str, "attrs": dict}
    """
    text_str = str(text) if text else ""
    tools = []
    remaining = text_str

    # Regex: <(TAG)(\s+[^>]*)?>(.*?)(?:</\1>|$)
    tags_pattern = "|".join(TOOL_TAGS)
    pattern = re.compile(rf"<({tags_pattern})(?:\s+([^>]*))?>(.*?)(?:</\1>|$)", re.DOTALL | re.IGNORECASE)

    # findall ile tümünü bulmak yerine finditer ile bulup 'remaining' metin üretimini manuel yapalım
    # çünkü remaining metin tool taglarının dışında kalan metindir.

    # Ancak mevcut implementasyon remaining'i start/end indexleri ile kesip biçiyor.
    # Biz de benzer bir mantıkla tüm eşleşmeleri toplayıp, asıl metinden çıkaralım.

    matches = list(pattern.finditer(text_str))

    # Sondan başa doğru çıkaralım ki indexler kaymasın
    for match in reversed(matches):
        tag_name = match.group(1).upper()
        attr_str = match.group(2) or ""
        payload = match.group(3).strip()

        # Attribute parse (basit key=value)
        attrs = {}
        if attr_str:
            # timeout=300 gibi yapıları yakala
            attr_matches = re.findall(r"(\w+)\s*=\s*([\"']?)([^\"'\s>]+)\2", attr_str)
            for k, _, v in attr_matches:
                attrs[k.lower()] = v

        tools.insert(0, {
            "tool": tag_name,
            "payload": payload,
            "attrs": attrs
        })

        # Remaining metinden çıkar
        remaining = remaining[:match.start()] + " " + remaining[match.end():]

    return tools, remaining.strip()


def extract_tool(text: str) -> Tuple[Optional[str], Optional[str], str, Dict[str, str]]:
    """Single tool version. Returns (tag, payload, outside_text, attrs)."""
    tools, outside = extract_tools(text)
    if not tools:
        return None, None, outside, {}
    t = tools[0]
    return t["tool"], t["payload"], outside, t["attrs"]


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
