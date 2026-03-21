import logging
from typing import Dict, Any, List
import json

from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from bio_ml_agent.llm_backend import auto_create_backend

log = logging.getLogger("biology.wetlab")

class WetLabProtocolAgent(BaseSubAgent):
    """
    C6: Wet-Lab Protocol Agent
    Islak laboratuvar görevlerinde karmaşık makalelerden ve dokümanlardan
    çalışılabilir, adım adım test ve protokol metinlerini tasarlayan ajandır.
    
    Yetenekleri:
    - Protocol Summarization & Step Extraction
    - Reagent (Reaktif/Kimyasal) & Timing (Zamanlama) tabloları
    - Risk Points (Tehlike noktaları) ve Güvenlik Uyarıları
    - Troubleshooting notes (Olası hata senaryoları ve düzeltmeler)
    """

    def __init__(self, model_name: str = "gemini-2.5-pro"):
        super().__init__("WetLabProtocolAgent", model_name)
        self.llm = auto_create_backend(model_name)
        self.literature_context: str = ""
        self.protocol_evaluation: Dict[str, Any] = {}

    def perceive(self, context: Dict[str, Any]) -> None:
        """Makale özetleri veya metodoloji metinlerini (Methodology sections) alır."""
        self.literature_context = context.get("methodology_text", "")
        if not self.literature_context:
            log.warning("No methodology text provided to WetLabProtocolAgent.")
        else:
            log.info(f"🧪 WetLabAgent received literature/text context of length: {len(self.literature_context)}")

    def plan(self, goal: str) -> List[str]:
        return [
            "Extract actionable wet-lab procedures and tables (Steps, Reagents, Risks)."
        ]

    def act(self, step: str) -> Any:
        prompt = f"""
        You are an elite Wet-Lab Manager and Experimental Protocol Designer.
        Analyze the following academic methodology or experiment description.
        
        Literature/Methodology Source:
        {self.literature_context}
        
        Provide a highly sterile, organized, and actionable wet-lab protocol.
        Respond STRICTLY with a JSON dictionary containing your protocol synthesis:
        {{
            "protocol_summary_and_steps": ["step 1...", "step 2..."],
            "reagent_table": {{"chemicals": "volumes/concentrations..."}},
            "timing_table": {{"step": "duration..."}},
            "critical_risk_points": ["risk 1...", "risk 2..."],
            "troubleshooting_notes": "what to do if X fails..."
        }}
        Do NOT write navigational text. Only return the JSON (optionally wrapped in ```json markdown).
        """
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response_text = self.llm.chat(messages)
            clean_text = response_text.replace("```json", "").replace("```", "").strip()
            self.protocol_evaluation = json.loads(clean_text)
        except Exception as e:
            log.warning(f"WetLab parsing fell back to strings. Error: {e}")
            self.protocol_evaluation["unformatted_evaluation"] = response_text
            
        return f"Completed Protocol Synthesis"

    def verify(self, action_result: Any) -> bool:
        keys = ["protocol_summary_and_steps", "reagent_table"]
        return any(k in self.protocol_evaluation for k in keys)

    def summarize(self) -> AgentResult:
        is_valid = self.verify("")
        conf = Confidence.HIGH if is_valid else Confidence.LOW
        
        return AgentResult(
            success=is_valid,
            data={"wetlab_protocol": self.protocol_evaluation},
            confidence=conf,
            evidence=[Evidence(source="wetlab_engine", content_snippet="Protocol stractured.")],
            message=f"Wet-lab protocol synthesis complete."
        )
