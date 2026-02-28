"""Bioinformatician sub-agent for the Swarm Architecture."""
import logging
import re
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class BioinfoExpertAgent:
    def __init__(self, context):
        self.context = context
        self.system_prompt = (
            "Sen Bio-ML Swarm Topluluğunun 'Biyoinformatik Uzmanı'sın.\n"
            "Görevin: Medikal/Biyolojik verileri analiz etmektir. PDB dosyalarını okuma, protein dizilerini "
            "hizalama, GC içeriği, moleküler hidrofobisite veya Lipinski kuralı analizi yapabilirsin.\n\n"
            "Araçların: Sadece `bioeng_toolkit.py` içerisindeki `ProteinAnalyzer`, `GenomicAnalyzer` "
            "ve `DrugMolecule` sınıflarını <PYTHON>...</PYTHON> kod blokları ile kullanabilirsin.\n"
            "Sonuca her zaman biyolojik anlamlarını ekleyerek yanıt üret."
        )
        
    def execute(self) -> str:
        """Bioinformatician LLM zincirini başlatır."""
        from llm_backend import auto_create_backend
        from agent import extract_tools, run_python
        from progress import Spinner
        
        backend = auto_create_backend(self.context.model)
        
        messages = [{"role": "system", "content": self.system_prompt}]
        
        if self.context.history:
            messages.append(self.context.history[-1])
        
        logger.info("[Bioinformatician] Biyoinformatik görevine başlanıyor...")
        
        max_steps = 10
        final_answer = ""
        
        for step in range(max_steps):
            with Spinner(f"🧠 Biyoinformatik Uzmanı Düşünüyor (Adım {step+1}/{max_steps})"):
                response = backend.chat(messages)
            
            tools_to_run, outside = extract_tools(response)
            
            if not tools_to_run:
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
                    with Spinner("🐍 Biyoinformatik Python Çalıştırıyor"):
                        out = run_python(payload, py_cwd, timeout_s=120)
                    
                    formatted_out = f"\n🛠️ PYTHON output:\n{out}\n"
                    all_outputs.append(formatted_out)
                    print(formatted_out)
                else:
                    all_outputs.append(f"[BLOCKED] Sadece PYTHON aracı kullanabilirsin.")
            
            messages.append({"role": "user", "content": "\n".join(all_outputs)})
        
        self.context.shared_memory["bioinfo_last_status"] = "Biyoinformatik analizi tamamlandı."
        return final_answer if final_answer else "Biyoinformatik Uzmanı döngüsü sona erdi."
