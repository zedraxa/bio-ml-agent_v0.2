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
from typing import Optional, Any, List, Dict
try:
    from ultra_agent.observability.audit_trail import AuditTrailLogger
except ImportError:
    # Fallback for linter or missing module
    class AuditTrailLogger:
        def __init__(self, *args, **kwargs): pass
        def log_critical_action(self, *args, **kwargs): pass

log = logging.getLogger("browser_agent")


BROWSER_AGENT_SYSTEM = """Sen bir Browser Sub-Agent'sın. Headless bir tarayıcıyı kontrol ederek görevleri yerine getirirsin.

Her adımda sana:
1. Görev açıklaması
2. Sayfa özeti (Başlık, URL)
3. Etkileşimli elemanlar listesi (bio-id, rol, etiket vb.)
4. Yakın geçmiş (son 5 adım)
verilecek.

Senin görevin, bir sonraki adımı belirlemek ve SADECE aşağıdaki JSON formatında yanıt vermektir:

```json
{
  "thought": "Düşünce sürecin ve planın",
  "action": {
    "type": "goto|click|fill|press|select|scroll|wait_for|extract_text|screenshot|done|fail",
    "target": {"bio_id": "bio-1", "selector": "opsiyonel"},
    "value": "değer (gerekiyorsa)",
    "reason": "Kısa açıklama"
  }
}
```

Aksiyon Tipleri ve Kurallar:
- goto: Belirtilen URL'ye gider. (value: url)
- click: Belirtilen bio_id'li elemana tıklar. (target.bio_id: bio-X)
- fill: Belirtilen bio_id'li alana metin yazar. (target.bio_id: bio-X, value: metin)
- press: Bir tuşa basar (Enter, Tab vb.). (value: tuş adı)
- select: Dropdown'dan seçenek seçer. (target.bio_id: bio-X, value: seçenek)
- scroll: Sayfayı aşağı/yukarı kaydırır. (value: 'up'/'down'/'top'/'bottom')
- wait_for: Belirli bir süre bekler. (value: ms)
- extract_text: Eleman metnini okur. (target.bio_id: bio-X)
- screenshot: Ekran görüntüsü alır. (target.bio_id: bio-X opsiyonel)
- done: Görev bitti. (value: özet)
- fail: Hata oluştu. (value: neden)

Dikkat:
- Sadece geçerli JSON döndür, açıklama veya giriş metni ekleme.
- Elemanları bio-id'leri üzerinden hedefle.
"""


class BrowserSubAgent:
    """LLM-powered otonom browser agent."""

    def __init__(self, model: Optional[str] = None, workspace: Optional[Path] = None, project_name: str = "scratch_project", session_id: str = "default", max_steps: int = 15):
        self.model = model
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
            from llm_backend import auto_create_backend
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

                # Kullanıcı mesajını oluştur
                recent_history: List[str] = []
                count = len(results)
                for i in range(max(0, count-5), count):
                    recent_history.append(str(results[i]))
                        
                step_prompt = f"""Adım {step}/{self.max_steps}

GÖREV: {task}

SAYFA DURUMU:
- Title: {page_title}
- URL: {page_url}

ETKİLEŞİMLİ ELEMANLAR (Action Candidates):
{elements_list_str}

GEÇMIŞ ADIMLAR:
{chr(10).join(recent_history) if recent_history else "(henüz adım yok)"}

Şimdi ne yapmalısın? Bir sonraki komutu ver."""

                messages.append({"role": "user", "content": step_prompt})

                # LLM'den yanıt al
                try:
                    backend = auto_create_backend(self.model or "gemini-2.0-flash", mode="auto")
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
                    from ultra_agent.runtime.browser.dom_intelligence import LocatorResolver, ActionValidator
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
                        audit_logger.log_critical_action("BROWSER_DONE", "sub-agent", {"result": a_value}, "COMPLETED")
                        return f"[BAŞARILI] {a_value}"
                    
                    elif a_type == "fail":
                        audit_logger.log_critical_action("BROWSER_FAIL", "sub-agent", {"reason": a_value}, "FAILED")
                        return f"[HATA] {a_value}"

                    elif a_type == "goto":
                        audit_logger.log_critical_action("BROWSER_GOTO", "sub-agent", {"url": a_value}, "EXECUTED")
                        page.goto(a_value, wait_until="domcontentloaded", timeout=30000)
                        page.wait_for_timeout(1500)

                    elif a_type == "click":
                        if not target_locator: raise ValueError("Click için hedef gerekli.")
                        audit_logger.log_critical_action("BROWSER_CLICK", "sub-agent", {"target": target_selector_str}, "EXECUTED")
                        target_locator.first.click(timeout=10000)
                        page.wait_for_timeout(1000)

                    elif a_type == "fill":
                        if not target_locator: raise ValueError("Fill için hedef gerekli.")
                        audit_logger.log_critical_action("BROWSER_FILL", "sub-agent", {"target": target_selector_str}, "EXECUTED")
                        target_locator.first.fill(a_value, timeout=10000)

                    elif a_type == "press":
                        audit_logger.log_critical_action("BROWSER_PRESS", "sub-agent", {"key": a_value}, "EXECUTED")
                        if target_locator:
                            target_locator.first.press(a_value)
                        else:
                            page.keyboard.press(a_value)
                        page.wait_for_timeout(500)

                    elif a_type == "select":
                        if not target_locator: raise ValueError("Select için hedef gerekli.")
                        audit_logger.log_critical_action("BROWSER_SELECT", "sub-agent", {"target": target_selector_str, "value": a_value}, "EXECUTED")
                        target_locator.first.select_option(a_value)

                    elif a_type == "scroll":
                        direction = a_value.lower()
                        audit_logger.log_critical_action("BROWSER_SCROLL", "sub-agent", {"direction": direction}, "EXECUTED")
                        if "down" in direction: page.mouse.wheel(0, 500)
                        elif "up" in direction: page.mouse.wheel(0, -500)
                        elif "bottom" in direction: page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                        elif "top" in direction: page.evaluate("window.scrollTo(0, 0)")
                        page.wait_for_timeout(500)

                    elif a_type == "wait_for":
                        ms = int(a_value) if a_value.isdigit() else 2000
                        page.wait_for_timeout(ms)

                    elif a_type == "extract_text":
                        if not target_selector_str: target_selector_str = "body"
                        text_content = page.locator(target_selector_str).first.inner_text()
                        results.append(f"[{step}] 📄 Okunan Metin: {text_content[:200]}...")

                    elif a_type == "screenshot":
                        results.append(f"[{step}] 📸 Ekran görüntüsü teyidi.")

                    else:
                        results.append(f"[{step}] ⚠️ Bilinmeyen aksiyon tipi: {a_type}")
                        
                    # P7: After Screenshot ve Trace Kaydı
                    try:
                        page.wait_for_load_state("domcontentloaded", timeout=5000)
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
                    err = f"[{step}] ❌ Aksiyon hatası ({a_type}): {str(e)[:100]}"
                    results.append(err)
                    log.error(err)

        except Exception as e:
            log.error("🌐 BrowserSubAgent HATA: %s", e, exc_info=True)
            results.append(f"\n❌ Browser hatası: {type(e).__name__}: {str(e)[:300]}")

        elapsed = time.time() - start_time
        log.info("🌐 BrowserSubAgent tamamlandı | süre=%.2fs | adım=%d", elapsed, len(results))

        output = "\n".join(results)
        return f"[BROWSER_AGENT] {len(results)} adım | {elapsed:.1f}s\n\n{output}"


def run_browser_agent(task: str, model: Optional[str] = None, workspace: Optional[Path] = None, timeout_s: int = 180, project_name: str = "scratch_project", session_id: str = "default") -> str:
    """Legacy helper — BrowserWorker kullanarak çalıştır."""
    from ultra_agent.runtime.browser.browser_worker import BrowserWorker
    worker = BrowserWorker(workspace=workspace, project_name=project_name, session_id=session_id)
    return worker.run_task(task, model=model, timeout_s=timeout_s)
