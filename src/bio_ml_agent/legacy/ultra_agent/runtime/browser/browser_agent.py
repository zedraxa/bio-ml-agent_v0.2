"""
Browser Sub-Agent — Otonom Headless Browser Kontrolü
Agent'a bir görev ver, o kendi başına tarayıcıyı açıp düşünerek hareket eder.
Antigravity'nin browser_subagent'ı gibi çalışır.
"""

import re
import time
import json
import logging
import random
from pathlib import Path
from typing import Optional, Any, List, Dict
try:
    from bio_ml_agent.ultra_agent.observability.audit_trail import AuditTrailLogger
except ImportError:
    # Fallback for linter or missing module
    class AuditTrailLogger:
        def __init__(self, *args, **kwargs): pass
        def log_critical_action(self, *args, **kwargs): pass

log = logging.getLogger("browser_agent")


BROWSER_AGENT_SYSTEM = """Sen bir Browser Sub-Agent'sın. Headless veya normal bir tarayıcıyı kontrol ederek karmaşık web görevlerini (kayıt olma, veri toplama, arama, form doldurma vb.) yerine getirirsin.

Her adımda sana şunlar verilecek:
1. Görev açıklaması
2. Sayfa özeti (Başlık, URL)
3. Sayfa metin içeriği (sayfada görünen metinler — bunu DİKKATLİCE OKU)
4. Etkileşimli elemanlar listesi (bio-id, rol, etiket — Shadow DOM ve Iframe destekli)
5. Ekran görüntüsü (sayfanın görsel hali)
6. Son adım geçmişi

KRİTİK KURALLAR:
- Her adımda önce sayfa metnini OKU ve ANLA. Ne yazdığını bilerek hareket et.
- Screenshot'ı incele — sayfa gerçekten beklediğin gibi mi? "Robot musunuz?", "I am human", "Access Denied" veya CAPTCHA gibi bot engelleri var mı?
- Eğer bir bot kontrolü (CAPTCHA, Turnstile vb.) görüyorsan, onu geçmek için gerekli kutucuğa tıklamayı veya talimatları izlemeyi dene.
- Eleman listesinde hedefini bulamıyorsan, SCROLL yap veya sayfanın tamamen yüklenmesini bekle (wait_for).
- Bir aksiyon 2 kez üst üste başarısız oluyorsa, alternatif bir strateji geliştir (farklı buton, farklı seçici, geri dön).
- Sıkıştığını hissedersen "fail" ile durma, önce "back" ile geri dönmeyi veya farklı yaklaşımı dene.

DÜŞÜNCE SÜRECİ (Chain of Thought):
"thought" alanında şu adımları izle:
1. Şu an neredeyim? (URL, sayfa başlığı, görünen içerik)
2. Hedefime ulaşmak için ne yapmalıyım?
3. Hangi elemanı kullanmalıyım ve neden?
4. Risk var mı? (yanlış butona tıklama, yanlış sayfa, vs.)

Yanıtını SADECE aşağıdaki JSON formatında ver:

```json
{
  "thought": "Detaylı düşünce süreci — sayfa metnini okuyarak mevcut durumu analiz et, planını açıkla",
  "action": {
    "type": "goto|click|fill|press|select|scroll|wait_for|extract_text|screenshot|back|done|fail",
    "target": {"bio_id": "bio-1", "selector": "opsiyonel CSS selector"},
    "value": "değer (URL, metin, süre ms, vb.)",
    "reason": "Bu aksiyonu neden yaptığının kısa açıklaması"
  }
}
```

Aksiyon Tipleri:
- goto: URL'ye gider (value: tam URL)
- click: bio_id'li elemana tıklar
- fill: bio_id'li alana metin yazar (value: metin)
- press: Klavye tuşuna basar (value: Enter, Tab, Escape vb.)
- select: Dropdown'dan seçim yapar (value: seçenek değeri)
- scroll: Sayfayı kaydırır (value: down/up/bottom/top)
- wait_for: Bekler (value: milisaniye)
- extract_text: Seçili elemandan metin çeker
- screenshot: Ekran görüntüsü alır
- back: Bir önceki sayfaya geri döner (hata kurtarma için)
- done: Görev başarıyla tamamlandı (value: sonuç özeti)
- fail: Görev imkansız (value: detaylı neden)

Kural: SADECE geçerli JSON döndür. Açıklama veya markdown ekleme.
"""


class BrowserSubAgent:
    """LLM-powered otonom browser agent."""

    def __init__(self, model: Optional[str] = None, workspace: Optional[Path] = None, project_name: str = "scratch_project", session_id: str = "default", max_steps: int = 20):
        self.model = model or "gemini-2.0-flash"  # UP-3: Daha güçlü ve stabil model
        self.workspace = workspace or Path(".")
        self.project_name = project_name
        self.session_id = session_id
        self.max_steps = max_steps
        self.history = []

        # Artifact dizini (Varsayılan yapı, execute içinde override edilebilir)
        self.artifact_dir = self.workspace / self.project_name / "browser" / self.session_id
        self.action_log_file = self.artifact_dir / "actions.jsonl"

        # DOM Distiller scriptini yükle
        distiller_path = Path(__file__).parent / "distiller.js"
        if distiller_path.exists():
            self.distiller_js = distiller_path.read_text(encoding="utf-8")
        else:
            self.distiller_js = "return {page: {title: document.title, url: location.href}, interactive: []}"

    def execute(self, task: str, page: Any) -> str:
        """
        Görevi otonom olarak belirtilen (isolated) tarayıcı sayfasında çalıştırır.
        """
        log.info("🌐 BrowserSubAgent döngüsü başlatıldı | görev='%s'", str(task)[:100])

        # P4 -> P7 Uyumluluğu: Eğer BrowserWorker job dizinini page objesine eklerse onu kullan
        if hasattr(page, "_job_dir"):
             self.artifact_dir = getattr(page, "_job_dir")
             self.action_log_file = self.artifact_dir / "actions.jsonl"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        try:
            from bio_ml_agent.llm_backend import auto_create_backend
        except ImportError:
            return "[BROWSER_AGENT HATA] LLM backend import edilemedi."

        results: List[Any] = []
        start_time = time.time()

        messages = [
            {"role": "system", "content": BROWSER_AGENT_SYSTEM},
        ]

        try:
            for step in range(1, self.max_steps + 1):
                # P7 Adım dizini
                step_dir = self.artifact_dir / "steps" / f"{step:02d}_pending"
                step_dir.mkdir(parents=True, exist_ok=True)

                # 1. Before Screenshot
                try:
                    page.wait_for_load_state("domcontentloaded", timeout=5000)
                    page.screenshot(path=str(step_dir / "before.png"))
                except Exception as e:
                    log.debug("📸 Before screenshot alınamadı: %s", e)

                # 2. Perception (Distiller + Candidates Screenshot)
                perception: dict = {"page": {"title": "(bilinmiyor)", "url": "about:blank"}, "interactive": []}
                try:
                    eval_result = page.evaluate(self.distiller_js)
                    if isinstance(eval_result, dict):
                        perception = eval_result

                    # Candidates Screenshot (Overlay varken)
                    page.screenshot(path=str(step_dir / "candidates.png"))

                    # P7: Aksiyona hazırlanmak için overlay'i temizle
                    page.evaluate("document.getElementById('bio-ml-overlay')?.remove();")
                except Exception as e:
                    log.warning("🌐 DOM Distillation hatası: %s", e)
                    perception["page"] = {
                        "title": page.title() or "(hata)",
                        "url": page.url or "about:blank"
                    }

                # 3. DOM Snapshot
                try:
                    with open(step_dir / "dom_snapshot.html", "w", encoding="utf-8") as f:
                        f.write(page.content())
                except Exception:
                    pass

                page_info: Dict[str, Any] = perception.get("page", {})
                page_title = str(page_info.get("title", "(bilinmiyor)"))
                page_url = str(page_info.get("url", "about:blank"))
                interactive_elements: List[Any] = perception.get("interactive", [])

                # Elemanları özetle
                elements_summary: List[str] = []
                if isinstance(interactive_elements, list):
                    limit = min(50, len(interactive_elements))
                    for i in range(limit):
                        el = interactive_elements[i]
                        if not isinstance(el, dict): continue
                        bio_id = el.get('bio_id', 'unknown')
                        tag = el.get('tag', 'unknown')
                        summary = f"- [{bio_id}] {tag}"
                        if el.get('role'): summary += f" (role: {el['role']})"
                        if el.get('label'): summary += f" label: \"{el['label']}\""
                        if el.get('placeholder'): summary += f" placeholder: \"{el['placeholder']}\""
                        elements_summary.append(summary)

                elements_list_str = "\n".join(elements_summary) if elements_summary else "(etkileşimli eleman bulunamadı)"

                # UP-1: Sayfa metin içeriğini çek
                try:
                    page_text = page.inner_text("body")
                    # Fazla boşlukları temizle, max 3000 karakter
                    import re as _re
                    page_text = _re.sub(r'\n{3,}', '\n\n', page_text).strip()
                    if len(page_text) > 3000:
                        page_text = page_text[:3000] + "\n[...SAYFA METNİ KISILDI...]"
                except Exception:
                    page_text = "(sayfa metni okunamadı)"

                # UP-4: Geçmişi genişlet (5 → 15)
                recent_history: List[str] = []
                count = len(results)
                for i in range(max(0, count-15), count):
                    recent_history.append(str(results[i]))

                # UP-5: Hata takibi — art arda hata sayısını belirle
                consecutive_errors = 0
                for r in reversed(results):
                    if "❌" in str(r) or "⚠️" in str(r):
                        consecutive_errors += 1
                    else:
                        break
                stuck_warning = ""
                if consecutive_errors >= 2:
                    stuck_warning = f"\n⚠️ UYARI: Art arda {consecutive_errors} hata oluştu! Alternatif bir strateji dene: geri dön (back), farklı eleman seç, veya scroll yap."

                step_prompt = f"""Adım {step}/{self.max_steps}

GÖREV: {task}

SAYFA DURUMU:
- Title: {page_title}
- URL: {page_url}

SAYFA METİN İÇERİĞİ (sayfada görünen yazılar):
{page_text}

ETKİLEŞİMLİ ELEMANLAR (Action Candidates):
{elements_list_str}

GEÇMİŞ ADIMLAR (son {min(15, count)}):
{chr(10).join(recent_history) if recent_history else "(henüz adım yok)"}
{stuck_warning}
Şimdi ne yapmalısın? Düşünceni detaylıca yaz, sonra bir sonraki komutu JSON olarak ver."""

                # UP-2: Multimodal — screenshot'ı mesaja ekle
                screenshot_path = str(step_dir / "before.png")
                if Path(screenshot_path).exists():
                    step_message_content = [
                        {"type": "text", "text": step_prompt},
                        {"type": "file", "path": screenshot_path}
                    ]
                else:
                    step_message_content = step_prompt

                messages.append({"role": "user", "content": step_message_content})

                # UP-3: Upgraded model
                try:
                    backend = auto_create_backend(self.model, mode="auto")
                    llm_response = backend.chat(messages)
                except Exception as e:
                    results.append(f"[{step}] ❌ LLM hatası: {e}")
                    break

                messages.append({"role": "assistant", "content": llm_response})

                # JSON Parse Et (P3)
                try:
                    clean_json = llm_response.strip()
                    if "```json" in clean_json:
                        clean_json = clean_json.split("```json")[1].split("```")[0].strip()
                    elif "```" in clean_json:
                        clean_json = clean_json.split("```")[1].split("```")[0].strip()

                    clean_json = re.sub(r"</?CMD>", "", clean_json).strip()

                    response_data = json.loads(clean_json)
                    thought = response_data.get("thought", "(düşünce belirtilmedi)")
                    action = response_data.get("action", {})
                    a_type = action.get("type", "unknown").lower()
                    a_target = action.get("target", {})
                    a_bio_id = a_target.get("bio_id")
                    a_selector = a_target.get("selector")
                    a_value = str(action.get("value", ""))
                    a_reason = action.get("reason", "")

                    log.info(f"🤖 Step {step} | Thought: {thought[:100]}")
                    log.info(f"🤖 Action: {a_type} | Target: {a_bio_id or a_selector} | Reason: {a_reason}")
                    results.append(f"[{step}] {a_type}: {a_bio_id or a_selector or a_value}")

                    # P7: Adım klasörünü aksiyon adıyla güncelle
                    import shutil
                    new_step_dir = step_dir.parent / f"{step:02d}_{a_type}"
                    if step_dir.exists() and not new_step_dir.exists():
                        step_dir.rename(new_step_dir)
                        step_dir = new_step_dir

                except Exception as e:
                    error_msg = f"[{step}] ❌ JSON Parsing Hatası: {e}"
                    results.append(error_msg)
                    log.warning(error_msg)
                    continue

                # Aksiyonu Uygula (P3 + P6)
                try:
                    audit_logger = AuditTrailLogger(self.workspace / self.project_name)

                    # P6: Çözümleyici ve Doğrulayıcı
                    from bio_ml_agent.ultra_agent.runtime.browser.dom_intelligence import LocatorResolver, ActionValidator
                    resolver = LocatorResolver(page, perception)

                    target_locator = None
                    target_selector_str = None # Audit log için fallback string

                    if a_bio_id and a_bio_id != "unknown":
                        target_locator = resolver.resolve(a_bio_id)
                        target_selector_str = f"[data-bio-id='{a_bio_id}']"
                    elif a_selector:
                        target_locator = page.locator(a_selector).first
                        target_selector_str = a_selector

                    # P6: Doğrulama (Sadece DOM ile etkileşenler için)
                    if target_locator and a_type not in ["done", "fail", "goto", "wait_for", "screenshot", "scroll"]:
                        validator = ActionValidator(target_locator, a_type)
                        is_valid, reason = validator.validate()
                        if not is_valid:
                            err_msg = f"[{step}] ⚠️ Aksiyon reddedildi: {reason}"
                            results.append(err_msg)
                            log.warning(err_msg)
                            continue

                    if a_type == "done":
                        audit_logger.log_critical_action("sub-agent", "BROWSER_DONE", {"result": a_value}, "COMPLETED")
                        return f"[BAŞARILI] {a_value}"

                    elif a_type == "fail":
                        audit_logger.log_critical_action("sub-agent", "BROWSER_FAIL", {"reason": a_value}, "FAILED")
                        return f"[HATA] {a_value}"

                    # UP-5: back aksiyonu — hata kurtarma
                    elif a_type == "back":
                        audit_logger.log_critical_action("sub-agent", "BROWSER_BACK", {"reason": a_value}, "EXECUTED")
                        page.go_back(wait_until="domcontentloaded", timeout=10000)
                        page.wait_for_timeout(1000)
                        results.append(f"[{step}] ◀️ Geri dönüldü: {a_value}")

                    # --- Retry gerektirmeyen aksiyonlar ---
                    elif a_type == "wait_for":
                        ms = int(a_value) if a_value.isdigit() else 2000
                        page.wait_for_timeout(ms)
                        results.append(f"[{step}] ⏳ {ms}ms beklendi")

                    elif a_type == "extract_text":
                        sel = target_selector_str or "body"
                        text_content = page.locator(sel).first.inner_text()
                        results.append(f"[{step}] 📄 Okunan Metin: {text_content[:200]}...")

                    elif a_type == "screenshot":
                        results.append(f"[{step}] 📸 Ekran görüntüsü teyidi.")

                    # --- Retry mekanizmalı aksiyonlar ---
                    elif a_type in ("goto", "click", "fill", "press", "select", "scroll"):
                        max_retries = 3
                        retry_count = 0
                        action_success = False
                        last_action_error = ""

                        while retry_count < max_retries and not action_success:
                            try:
                                if a_type == "goto":
                                    audit_logger.log_critical_action("sub-agent", "BROWSER_GOTO", {"url": a_value}, "EXECUTED")
                                    page.goto(a_value, wait_until="networkidle", timeout=30000)
                                    action_success = True

                                elif a_type == "click":
                                    if not target_locator: raise ValueError("Click için hedef gerekli.")
                                    audit_logger.log_critical_action("sub-agent", "BROWSER_CLICK", {"target": target_selector_str}, "EXECUTED")
                                    target_locator.first.click(force=True, timeout=10000)
                                    page.wait_for_load_state("domcontentloaded", timeout=5000)
                                    action_success = True

                                elif a_type == "fill":
                                    if not target_locator: raise ValueError("Fill için hedef gerekli.")
                                    audit_logger.log_critical_action("sub-agent", "BROWSER_FILL", {"target": target_selector_str}, "EXECUTED")
                                    target_locator.first.fill("", timeout=5000)
                                    target_locator.first.fill(a_value, timeout=5000)
                                    action_success = True

                                elif a_type == "press":
                                    audit_logger.log_critical_action("sub-agent", "BROWSER_PRESS", {"key": a_value}, "EXECUTED")
                                    if target_locator:
                                        target_locator.first.press(a_value, timeout=5000)
                                    else:
                                        page.keyboard.press(a_value)
                                    page.wait_for_load_state("domcontentloaded", timeout=5000)
                                    action_success = True

                                elif a_type == "select":
                                    if not target_locator: raise ValueError("Select için hedef gerekli.")
                                    audit_logger.log_critical_action("sub-agent", "BROWSER_SELECT", {"target": target_selector_str, "value": a_value}, "EXECUTED")
                                    target_locator.first.select_option(a_value, timeout=5000)
                                    action_success = True

                                elif a_type == "scroll":
                                    direction = a_value.lower()
                                    audit_logger.log_critical_action("sub-agent", "BROWSER_SCROLL", {"direction": direction}, "EXECUTED")
                                    if "down" in direction: page.mouse.wheel(0, 500)
                                    elif "up" in direction: page.mouse.wheel(0, -500)
                                    elif "bottom" in direction: page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                                    elif "top" in direction: page.evaluate("window.scrollTo(0, 0)")
                                    page.wait_for_timeout(500)
                                    action_success = True

                            except Exception as retry_err:
                                retry_count += 1
                                last_action_error = str(retry_err)
                                log.debug(f"[Retry {retry_count}/{max_retries}] Action {a_type} failed: {last_action_error}")
                                page.wait_for_timeout(1000)

                        if not action_success:
                            # UP-5: Hata kurtarma — başarısız aksiyonu raporla ama döngüyü kırma
                            err_msg = f"[{step}] ❌ {a_type} başarısız ({max_retries} deneme): {last_action_error[:100]}"
                            results.append(err_msg)
                            log.warning(err_msg)

                    else:
                        results.append(f"[{step}] ⚠️ Bilinmeyen aksiyon tipi: {a_type}")

                    # P7: After Screenshot ve Trace Kaydı
                    try:
                        page.wait_for_load_state("domcontentloaded", timeout=3000)
                        page.screenshot(path=str(step_dir / "after.png"))
                    except Exception as e:
                        log.debug("📸 After screenshot alınamadı: %s", e)

                    step_meta = {
                        "step": step,
                        "timestamp": time.time(),
                        "thought": thought,
                        "action": action,
                        "resolved_target": target_selector_str,
                        "result_msg": results[-1] if results else "OK"
                    }
                    with open(step_dir / "step_meta.json", "w", encoding="utf-8") as f:
                        json.dump(step_meta, f, indent=2, ensure_ascii=False)

                    # P7: Global Action Log
                    with open(self.action_log_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(step_meta, ensure_ascii=False) + "\n")

                except Exception as e:
                    # HATA DURUMUNDA DA EKRAN GÖRÜNTÜSÜ AL
                    try:
                        page.screenshot(path=str(step_dir / "error.png"))
                    except:
                        pass
                    err = f"[{step}] ❌ Aksiyon hatası ({a_type}): {str(e)[:200]}"
                    results.append(err)
                    log.error(err, exc_info=True)

        except Exception as e:
            log.error("🌐 BrowserSubAgent HATA: %s", e, exc_info=True)
            results.append(f"\n❌ Browser hatası: {type(e).__name__}: {str(e)[:300]}")

        elapsed = time.time() - start_time
        log.info("🌐 BrowserSubAgent tamamlandı | süre=%.2fs | adım=%d", elapsed, len(results))

        output = "\n".join(results)
        return f"[BROWSER_AGENT] {len(results)} adım | {elapsed:.1f}s\n\n{output}"


def run_browser_agent(task: str, model: Optional[str] = None, workspace: Optional[Path] = None, timeout_s: int = 180, project_name: str = "scratch_project", session_id: str = "default") -> str:
    """Legacy helper — BrowserWorker kullanarak çalıştır."""
    from bio_ml_agent.ultra_agent.runtime.browser.browser_worker import BrowserWorker
    worker = BrowserWorker(workspace=workspace, project_name=project_name, session_id=session_id)
    return worker.run_task(task, model=model, timeout_s=timeout_s)
