"""
BrowserWorker — İzole Browser Bağlamı Yöneticisi (P4 + P5)

P4: Her görev için ayrı BrowserContext, trace, video, artifact yönetimi.
P5: Tenant/Profile Isolation — üç mod:
    ephemeral : Her job sıfır context (stateless)
    session   : Aynı proje/session içinde state korunur
    trusted   : Kayıtlı login state yüklenir + domain policy
"""

import logging
import time
import json
from pathlib import Path
from typing import Optional, Any, Dict
from playwright_stealth import stealth_sync

log = logging.getLogger("browser_worker")


class BrowserWorker:
    """
    Browser otomasyonu için izole worker.
    Politika dosyasına göre context izolasyonu sağlar.
    """

    def __init__(
        self,
        workspace: Optional[Path] = None,
        project_name: str = "scratch_project",
        session_id: str = "default",
        headless: bool = False,
        enable_video: bool = True,
        enable_tracing: bool = True,
        isolation_mode: str = "ephemeral",
    ):
        self.workspace = workspace or Path(".")
        self.project_name = project_name
        self.session_id = session_id
        self.headless = headless
        self.enable_video = enable_video
        self.enable_tracing = enable_tracing
        self.isolation_mode = isolation_mode

        # Artifact dizini
        self.base_dir = self.workspace / "browser_artifacts" / project_name / session_id
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Politikayı yükle
        self.policy = self._load_policy()

    # ─── Politika Yönetimi ───

    def _load_policy(self):
        """Politika dosyasını yükler; yoksa constructor parametrelerinden oluşturur."""
        from ultra_agent.runtime.browser.browser_policy import BrowserPolicy
        policy = BrowserPolicy.load(self.base_dir)

        # Constructor parametreleri ile override (ilk kez oluşturuluyor olabilir)
        if self.isolation_mode != "ephemeral" or not (self.base_dir / "policy.json").exists():
            policy.mode = self.isolation_mode
            policy.enable_video = self.enable_video
            policy.enable_tracing = self.enable_tracing
            policy.save(self.base_dir)

        # Worker seviyesi bayrakları politikadan senkronize et
        self.enable_video = policy.enable_video
        self.enable_tracing = policy.enable_tracing
        return policy

    # ─── Job Dizini ───

    def _create_job_dir(self) -> Path:
        """Her görev için benzersiz alt dizin."""
        job_id = f"job_{int(time.time())}"
        job_dir = self.base_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    # ─── Context Opsiyonları ───

    def _build_context_options(self, job_dir: Path) -> Dict[str, Any]:
        """Politikaya göre BrowserContext opsiyonlarını hazırlar."""
        from ultra_agent.runtime.browser.browser_policy import resolve_state_path

        ctx_opts: Dict[str, Any] = {
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36", # P2: Windows 10 Chrome (daha az bot şüphesi)
            "viewport": {"width": 1366, "height": 768},
            "device_scale_factor": 1,
            "has_touch": False,
            "is_mobile": False,
        }

        # Anti-Bot (Headless) args
        if getattr(self, "headless", True):
             ctx_opts["args"] = [
                 "--disable-blink-features=AutomationControlled", # En kritik bot koruma bayrağı bypass
                 "--no-sandbox",
                 "--disable-dev-shm-usage"
             ]

        # Video
        if self.policy.enable_video:
            video_dir = job_dir / "videos"
            video_dir.mkdir(parents=True, exist_ok=True)
            ctx_opts["record_video_dir"] = str(video_dir)

        # State yükle (session / trusted)
        if self.policy.should_load_state:
            state_path = resolve_state_path(self.policy, self.base_dir)
            if state_path and state_path.exists():
                log.info("🔑 Storage state yükleniyor: %s (mod=%s)", state_path, self.policy.mode)
                ctx_opts["storage_state"] = str(state_path)
            elif self.policy.is_trusted:
                log.warning("⚠️ Trusted mod ama state dosyası bulunamadı: %s", state_path)

        return ctx_opts

    # ─── State Kaydetme ───

    def _save_state(self, context: Any, job_dir: Path) -> Optional[Path]:
        """Session/trusted modda context state'ini diske kaydeder."""
        from ultra_agent.runtime.browser.browser_policy import resolve_state_path

        if not self.policy.should_persist_state:
            return None

        state_path = resolve_state_path(self.policy, self.base_dir)
        if not state_path:
            return None

        try:
            context.storage_state(path=str(state_path))
            log.info("💾 Storage state kaydedildi: %s", state_path)

            # Job dizinine de kopyala (snapshot)
            job_state = job_dir / "storage_state_snapshot.json"
            import shutil
            shutil.copy2(str(state_path), str(job_state))

            return state_path
        except Exception as e:
            log.warning("⚠️ State kaydetme hatası: %s", e)
            return None

    # ─── Domain Policy ───

    def _apply_domain_policy(self, page: Any) -> None:
        """Engellenecek domainleri page.route() ile bloke eder."""
        if not self.policy.blocked_domains and not self.policy.allowed_domains:
            return

        policy = self.policy

        def _route_handler(route: Any) -> None:
            url = route.request.url
            if not policy.is_domain_allowed(url):
                log.info("🚫 Domain engellendi: %s", url[:100])
                route.abort("blockedbyclient")
            else:
                route.continue_()

        try:
            page.route("**/*", _route_handler)
            log.info("🛡️ Domain politikası uygulandı | blocked=%d | allowed=%d",
                     len(policy.blocked_domains), len(policy.allowed_domains))
        except Exception as e:
            log.warning("⚠️ Domain politikası uygulanamadı: %s", e)

    # ─── Ana Çalıştırıcı ───

    def run_task(self, task: str, model: Optional[str] = None, timeout_s: int = 180) -> str:
        """Görevi izole bir context içinde çalıştırır."""
        log.info("🚀 BrowserWorker | project=%s | session=%s | mod=%s",
                 self.project_name, self.session_id, self.policy.mode)

        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return "[BROWSER_WORKER HATA] Playwright yüklü değil. Kur: pip install playwright && playwright install chromium"

        from ultra_agent.runtime.browser.browser_agent import BrowserSubAgent

        # Timeout'u politika ile sınırla
        effective_timeout = min(timeout_s, self.policy.max_timeout_s)

        start_time = time.time()
        job_dir = self._create_job_dir()
        result = "[HATA] Başlatılamadı"

        job_meta: Dict[str, Any] = {
            "task": task[:500],
            "project": self.project_name,
            "session": self.session_id,
            "isolation_mode": self.policy.mode,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "headless": self.headless,
            "video": self.policy.enable_video,
            "tracing": self.policy.enable_tracing,
            "effective_timeout": effective_timeout,
        }

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=self.headless)
                ctx_opts = self._build_context_options(job_dir)
                context = browser.new_context(**ctx_opts)

                # Tracing
                if self.policy.enable_tracing:
                    context.tracing.start(screenshots=True, snapshots=True, sources=True)

                try:
                    page = context.new_page()
                    
                    # P7: Stealth (Anti-Bot) Aktifleştir - (playwright-stealth)
                    stealth_sync(page)
                    
                    page.set_default_timeout(min(effective_timeout, 45) * 1000) # Timeout 60'tan 45'e düşürüldü - Çok uzun asılı kalmaları engelle
                    
                    # P7: Alt ajanın job dizinine kaydedebilmesi için page içerisine ekle
                    page._job_dir = job_dir

                    # Domain policy uygula
                    self._apply_domain_policy(page)

                    # Agent Execution
                    agent = BrowserSubAgent(
                        model=model,
                        workspace=self.workspace,
                        project_name=self.project_name,
                        session_id=self.session_id,
                    )

                    result = agent.execute(task, page)
                    job_meta["status"] = "completed"

                except Exception as e:
                    log.error("❌ BrowserWorker agent HATA: %s", e, exc_info=True)
                    result = f"[WORKER HATA] {type(e).__name__}: {e}"
                    job_meta["status"] = "error"
                    job_meta["error"] = str(e)[:500]

                finally:
                    # State kaydet (session/trusted)
                    saved_state = self._save_state(context, job_dir)
                    if saved_state:
                        job_meta["state_saved"] = str(saved_state)

                    # Tracing kaydet
                    if self.policy.enable_tracing:
                        trace_file = job_dir / "trace.zip"
                        try:
                            context.tracing.stop(path=str(trace_file))
                            log.info("📦 Trace kaydedildi: %s", trace_file)
                        except Exception as e:
                            log.warning("⚠️ Trace kaydı hatası: %s", e)

                    context.close()
                    browser.close()

        except Exception as e:
            log.error("❌ BrowserWorker Playwright HATA: %s", e, exc_info=True)
            result = f"[WORKER HATA] Playwright hatası: {type(e).__name__}: {e}"
            job_meta["status"] = "playwright_error"
            job_meta["error"] = str(e)[:500]

        elapsed = time.time() - start_time
        job_meta["elapsed_s"] = round(elapsed, 2)
        job_meta["result_preview"] = result[:300]

        # Metadata kaydet
        try:
            meta_file = job_dir / "job_meta.json"
            meta_file.write_text(json.dumps(job_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

        log.info("🏁 BrowserWorker tamamladı | süre=%.2fs | mod=%s | durum=%s",
                 elapsed, self.policy.mode, job_meta.get("status", "unknown"))

        return result
