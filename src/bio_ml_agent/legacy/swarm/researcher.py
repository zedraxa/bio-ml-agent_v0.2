"""Researcher sub-agent for the Swarm Architecture."""
import logging
import re
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

from .base import BaseAgent, SwarmContext

class ResearchAgent(BaseAgent):
    def __init__(self, context: SwarmContext):
        super().__init__(name="Researcher", role="Literature & Clinical Research", context=context)
        self.system_prompt = (
            "Sen Bio-ML Swarm Topluluğunun 'Araştırmacı' (Researcher) ajanısın.\n"
            "Görevin: Kullanıcının biyomedikal soruları veya veri setleri hakkında literatür taraması yapmak,\n"
            "klinik veritabanlarında (PubMed, ClinicalTrials.gov vb.) araştırma yapmak ve güncel bilgileri toplamaktır.\n\n"
            "Araçların:\n"
            "1. <WEB_SEARCH>aranacak kelime</WEB_SEARCH>: Hızlı bilgi toplama.\n"
            "2. <BROWSER_AGENT>detaylı araştırma talimatı</BROWSER_AGENT>: Karmaşık web sitelerinde gezinme ve veri çekme.\n\n"
            "DİKKAT: Sadece araştırma yaparsın. Kod yazmaz veya model eğitmezsin.\n"
            "Bulduğun bilgileri düzenli bir özet halinde sunmalısın.\n"
            "Özellikle 'Biyoinformatik Uzmanı'nın yorumlaması için klinik korelasyonlar bulmaya odaklan."
        )

    def get_system_prompt(self) -> str:
        return self.system_prompt

    def execute(self, task_prompt: str = "", error_history: str = "") -> str:
        """Researcher LLM zincirini başlatır."""
        from llm_backend import auto_create_backend
        from core.tools import extract_tools
        from progress import Spinner

        backend = auto_create_backend(self.context.model)
        messages = [{"role": "system", "content": self.system_prompt}]

        if task_prompt:
            messages.append({"role": "user", "content": task_prompt})

        logger.info("[Researcher] Araştırma görevine başlanıyor...")

        max_steps = 5
        final_answer = ""

        for step in range(max_steps):
            with Spinner(f"🧠 Researcher Araştırıyor (Adım {step+1}/{max_steps})"):
                response = backend.chat(messages)

            tools_to_run, outside = extract_tools(response)
            messages.append({"role": "assistant", "content": response})

            if not tools_to_run:
                final_answer = response
                break

            all_outputs = []
            for tool, payload in tools_to_run:
                if tool == "WEB_SEARCH":
                    from core.tools import web_search
                    out = web_search(payload)
                    all_outputs.append(f"\n🌐 WEB_SEARCH output:\n{out}\n")
                elif tool == "BROWSER_AGENT":
                    from browser_agent import run_browser_agent
                    out = run_browser_agent(payload, model=self.context.model, workspace=self.context.workspace)
                    all_outputs.append(f"\n🌐 BROWSER output:\n{out}\n")

            messages.append({"role": "user", "content": "\n".join(all_outputs)})

        return final_answer if final_answer else "Araştırma tamamlandı."
