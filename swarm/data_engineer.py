"""Data Engineer sub-agent for the Swarm Architecture."""
import logging
import re
import time
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

from .base import BaseAgent, SwarmContext

class DataEngineerAgent(BaseAgent):
    def __init__(self, context: SwarmContext):
        super().__init__(name="Data Engineer", role="Data Processing", context=context)
        self.system_prompt = (
            "Sen Bio-ML Swarm Topluluğunun 'Veri Mühendisi' (Data Engineer) ajanısın.\n"
            "Görevin: Veri setlerini indirmek, pandas ile incelemek, eksik verileri temizlemek "
            "ve özellikleri (features) ölçeklendirmek (Scaling/Encoding).\n"
            "Modelleri eğitmeyeceksin. Sadece veriyi ML Uzmanına hazır hale getireceksin.\n"
            "Workspace klasörüne '.csv' olarak temizlenmiş verileri kaydetmelisin.\n\n"
            "İşlemini tamamladığında her zaman \"Veri temizleme tamamlandı, dosya: X\" şeklinde final yanıtı ver.\n"
            "Araçların: Sadece <PYTHON>...</PYTHON> kod bloklarını kullanabilirsin.\n"
            "İnternetten bilgi bulman gerekirse ÖNCELİKLE <WEB_SEARCH>aranacak kelime</WEB_SEARCH> kullan (çok hızlıdır). Eğer basit arama yetersizse ve karmaşık bir sitede gezinmen/butonlara tıklaman/giriş yapman gerekirse <BROWSER_AGENT>talimat</BROWSER_AGENT> kullan.\n"        )
        
    def get_system_prompt(self) -> str:
        return self.system_prompt
        
    def execute(self, task_prompt: str = "", error_history: str = "") -> str:
        """Data Engineer LLM zincirini başlatır."""
        from llm_backend import auto_create_backend
        from agent import extract_tools, run_python
        from progress import Spinner
        
        backend = auto_create_backend(self.context.model)
        
        messages = [{"role": "system", "content": self.system_prompt}]
        
        if error_history:
            messages.append({"role": "system", "content": f"ÖNEMLİ HATA UYARISI: Önceki adımda şu hata alındı, lütfen veriyi düzeltip tekrar kaydet:\n{error_history}"})
            
        if task_prompt:
            messages.append({"role": "user", "content": task_prompt})
        elif self.context.history:
            # Enjecte edilen tarihçe, kullanıcı promptunu içerir
            messages.append(self.context.history[-1])
        
        logger.info("[Data Engineer] Veri işleme görevine başlanıyor...")
        
        max_steps = 10
        final_answer = ""
        
        for step in range(max_steps):
            with Spinner(f"🧠 Data Engineer Düşünüyor (Adım {step+1}/{max_steps})"):
                response = backend.chat(messages)
            
            tools_to_run, outside = extract_tools(response)
            
            # Fallback regex extraction for <PYTHON> if not caught by extract_tools standard format
            if not tools_to_run:
                import re
                py_m = re.search(r"<PYTHON>\s*(.*?)\s*</PYTHON>", response, re.DOTALL)
                if py_m:
                    tools_to_run = [("PYTHON", py_m.group(1))]
                else:
                    br_m = re.search(r"<BROWSER_AGENT>\s*(.*?)\s*</BROWSER_AGENT>", response, re.DOTALL)
                    if br_m:
                        tools_to_run = [("BROWSER_AGENT", br_m.group(1))]
                    else:
                        ws_m = re.search(r"<WEB_SEARCH>\s*(.*?)\s*</WEB_SEARCH>", response, re.DOTALL)
                        if ws_m:
                            tools_to_run = [("WEB_SEARCH", ws_m.group(1))]

            messages.append({"role": "assistant", "content": response})
            
            if not tools_to_run:
                final_answer = response
                break
                
            all_outputs = []
            for tool, payload in tools_to_run:
                if tool == "PYTHON":
                    from pathlib import Path
                    py_cwd = Path(self.context.workspace)
                    py_cwd.mkdir(parents=True, exist_ok=True)
                    with Spinner("🐍 Data Engineer Python Çalıştırıyor"):
                        out = run_python(payload, py_cwd, timeout_s=120)
                        
                    formatted_out = f"\\n🛠️ PYTHON output:\\n{out}\\n"
                    all_outputs.append(formatted_out)
                    print(formatted_out)
                elif tool == "BROWSER_AGENT":
                    from browser_agent import run_browser_agent
                    with Spinner("🌐 Data Engineer Browser Kullanıyor"):
                        out = run_browser_agent(payload, model=self.context.model, workspace=self.context.workspace)
                    formatted_out = f"\n🌐 BROWSER output:\n{out}\n"
                    all_outputs.append(formatted_out)
                    print(formatted_out)
                elif tool == "WEB_SEARCH":
                    from agent import web_search
                    with Spinner("🌐 Data Engineer Web Araştırması Yapıyor"):
                        try:
                            out = web_search(payload)
                        except Exception as e:
                            out = f"[ERROR] WEB_SEARCH hatası: {e}"
                    formatted_out = f"\n🌐 WEB_SEARCH output:\n{out}\n"
                    all_outputs.append(formatted_out)
                    print(formatted_out)
                else:
                    all_outputs.append(f"[BLOCKED] Sadece PYTHON, WEB_SEARCH ve BROWSER_AGENT kullanabilirsin.")
            
            messages.append({"role": "user", "content": "\\n".join(all_outputs)})
        
        self.context.shared_memory["data_engineer_last_status"] = "Veri işleme adımları tamamlandı."
        return final_answer if final_answer else "Veri Mühendisi döngüsü sona erdi."
