"""Bioinformatician sub-agent for the Swarm Architecture."""
import logging
import re
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

from .base import BaseAgent, SwarmContext

class BioinfoExpertAgent(BaseAgent):
    def __init__(self, context: SwarmContext):
        super().__init__(name="Bioinformatics Expert", role="Bioinformatics", context=context)
        self.system_prompt = (
            "Sen Bio-ML Swarm Topluluğunun 'Biyoinformatik Uzmanı'sın.\n"
            "Görevin: Medikal/Biyolojik verileri analiz etmektir. PDB dosyalarını okuma, protein dizilerini "
            "hizalama, GC içeriği, moleküler hidrofobisite veya Lipinski kuralı analizi yapabilirsin.\n\n"
            "Araçların: Sadece `bioeng_toolkit.py` içerisindeki `ProteinAnalyzer`, `GenomicAnalyzer` "
            "ve `DrugMolecule` sınıflarını <PYTHON>...</PYTHON> kod blokları ile kullanabilirsin. "
            "Ayrıca literatür taramak için <BROWSER_AGENT>ilgili araştırma konusu</BROWSER_AGENT> etiketini kullanabilirsin.\n"
            "ÖNEMLİ GÖREV: Sana bir 'Makine Öğrenimi (ML) Analiz Çıktısı' veya 'SHAP/LIME Feature Importance' verisi geldiğinde, "
            "bu özellikleri alıp TIBBİ ve BİYOLOJİK olarak yorumlamalısın (Örneğin, Vücut Kitle İndeksi neden diyabeti etkiler?). "
            "Format olarak raporunda '## Klinik Karar Özeti (XAI Yorumlaması)' başlığı altında detaylı analiz sunmalısın.\n"
            "Sonuca her zaman biyolojik anlamlarını ekleyerek kapsamlı bir yanıt üret."
        )

    def get_system_prompt(self) -> str:
        return self.system_prompt

    def execute(self, task_prompt: str = "", error_history: str = "") -> str:
        """Bioinformatician LLM zincirini başlatır."""
        from llm_backend import auto_create_backend
        from core.tools import extract_tools, run_python
        from progress import Spinner

        backend = auto_create_backend(self.context.model)

        messages = [{"role": "system", "content": self.system_prompt}]

        if error_history:
            messages.append({"role": "system", "content": f"ÖNEMLİ HATA UYARISI: Önceki denemede hata alındı. Lütfen düzelt:\n{error_history}"})

        if task_prompt:
            messages.append({"role": "user", "content": task_prompt})
        elif self.context.history:
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
                    with Spinner("🐍 Biyoinformatik Python Çalıştırıyor"):
                        out = run_python(payload, py_cwd, timeout_s=120)

                    formatted_out = f"\n🛠️ PYTHON output:\n{out}\n"
                    all_outputs.append(formatted_out)
                    print(formatted_out)
                elif tool == "BROWSER_AGENT":
                    from browser_agent import run_browser_agent
                    with Spinner("🌐 Biyoinformatik Uzmanı Browser Kullanıyor"):
                        out = run_browser_agent(payload, model=self.context.model, workspace=self.context.workspace)
                    formatted_out = f"\n🌐 BROWSER output:\n{out}\n"
                    all_outputs.append(formatted_out)
                    print(formatted_out)
                elif tool == "WEB_SEARCH":
                    from core.tools import web_search
                    with Spinner("🌐 Biyoinformatik Uzmanı Web Araştırması Yapıyor"):
                        try:
                            out = web_search(payload)
                        except Exception as e:
                            out = f"[ERROR] WEB_SEARCH hatası: {e}"
                    formatted_out = f"\n🌐 WEB_SEARCH output:\n{out}\n"
                    all_outputs.append(formatted_out)
                    print(formatted_out)
                else:
                    all_outputs.append(f"[BLOCKED] Sadece PYTHON, WEB_SEARCH ve BROWSER_AGENT aracı kullanabilirsin.")

            messages.append({"role": "user", "content": "\n".join(all_outputs)})

        self.context.shared_memory["bioinfo_last_status"] = "Biyoinformatik analizi tamamlandı."
        return final_answer if final_answer else "Biyoinformatik Uzmanı döngüsü sona erdi."
