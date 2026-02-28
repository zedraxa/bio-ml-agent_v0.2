"""Data Engineer sub-agent for the Swarm Architecture."""
import logging
import re
import time
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class DataEngineerAgent:
    def __init__(self, context):
        self.context = context
        self.system_prompt = (
            "Sen Bio-ML Swarm Topluluğunun 'Veri Mühendisi' (Data Engineer) ajanısın.\n"
            "Görevin: Veri setlerini indirmek, pandas ile incelemek, eksik verileri temizlemek "
            "ve özellikleri (features) ölçeklendirmek (Scaling/Encoding).\n"
            "Modelleri eğitmeyeceksin. Sadece veriyi ML Uzmanına hazır hale getireceksin.\n"
            "Workspace klasörüne '.csv' olarak temizlenmiş verileri kaydetmelisin.\n\n"
            "Araçların: Sadece <PYTHON>...</PYTHON> kodlarını kullanarak veri işleyebilirsin.\n"
        )
        
    def execute(self) -> str:
        """Data Engineer LLM zincirini başlatır."""
        from llm_backend import auto_create_backend
        from agent import extract_tools, run_python
        from progress import Spinner
        
        backend = auto_create_backend(self.context.model)
        
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Enjecte edilen tarihçe, kullanıcı promptunu içerir
        if self.context.history:
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
                else:
                    all_outputs.append(f"[BLOCKED] Data Engineer sadece PYTHON aracı kullanabilir.")
            
            messages.append({"role": "user", "content": "\\n".join(all_outputs)})
        
        self.context.shared_memory["data_engineer_last_status"] = "Veri işleme adımları tamamlandı."
        return final_answer if final_answer else "Veri Mühendisi döngüsü sona erdi."
