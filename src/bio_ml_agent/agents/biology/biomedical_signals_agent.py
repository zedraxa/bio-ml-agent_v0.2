import logging
from typing import Dict, Any, List
import json

from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from bio_ml_agent.llm_backend import auto_create_backend

log = logging.getLogger("biology.signals")

class BiomedicalSignalsAgent(BaseSubAgent):
    """
    C4: Biomedical Signals Agent
    EKG, EMG, EEG, PPG gibi elektriksel / optik zaman serisi biyomedikal
    sinyalleri yorumlayan uzman ajandır.
    
    Yetenekleri:
    - Sinyal Preprocessing & Filtreleme Önerileri (High-pass, band-stop/notch vs)
    - Peak / Event Detection (Örn: R-peak, Spindle) tespiti ve anlamlandırılması
    - Segmentation / Epoching mantığı
    - Feature Extraction yorumları (Time domain, Frequency domain)
    - Classification (Sınıflandırma) algoritmaları için baseline tasarım.
    """

    def __init__(self, model_name: str = "gemini-2.5-pro"):
        super().__init__("BiomedicalSignalsAgent", model_name)
        self.llm = auto_create_backend(model_name)
        self.signal_data: Dict[str, Any] = {}
        self.signal_evaluation: Dict[str, Any] = {}

    def perceive(self, context: Dict[str, Any]) -> None:
        """Sensörlerden, giyilebilir cihazlardan veya PhysioNet türevi dosyalardan (EDF vb.) gelen sinyal metriklerini alır."""
        self.signal_data = context.get("signal_data", {})
        if not self.signal_data:
            log.warning("No signal data provided to BiomedicalSignalsAgent.")
        else:
            log.info(f"📈 SignalAgent received datasets: {list(self.signal_data.keys())}")

    def plan(self, goal: str) -> List[str]:
        return [
            "Perform comprehensive Biomedical Signal analysis (Preprocessing, Feature Extraction, Event Detection, Classification)."
        ]

    def act(self, step: str) -> Any:
        prompt = f"""
        You are a Senior Biomedical Signal Processing Engineer.
        Analyze the following extracted physiological signal metrics (e.g. ECG, EEG, EMG, PPG).
        
        Signal Data / Meta:
        {json.dumps(self.signal_data, indent=2)}
        
        Provide a deep signal processing architecture and clinical interpretation.
        Respond STRICTLY with a JSON dictionary containing your assessment:
        {{
            "preprocessing_and_filtering_strategy": "interpretation (notch, bandpass etc)...",
            "peak_and_event_detection_logic": "interpretation...",
            "segmentation_and_epoching": "interpretation...",
            "feature_extraction_recommendation": "time/frequency domain interpretation...",
            "classification_baseline_architecture": "interpretation...",
            "signal_conclusion": "overall signal quality and clinical potential summary..."
        }}
        Do NOT write navigational text. Only return the JSON (optionally wrapped in ```json markdown).
        """
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response_text = self.llm.chat(messages)
            clean_text = response_text.replace("```json", "").replace("```", "").strip()
            self.signal_evaluation = json.loads(clean_text)
        except Exception as e:
            log.warning(f"Signal parsing fell back to strings. Error: {e}")
            self.signal_evaluation["unformatted_evaluation"] = response_text
            
        return f"Completed Signal Processing Evaluation"

    def verify(self, action_result: Any) -> bool:
        keys = ["preprocessing_and_filtering_strategy", "signal_conclusion"]
        return any(k in self.signal_evaluation for k in keys)

    def summarize(self) -> AgentResult:
        is_valid = self.verify("")
        conf = Confidence.HIGH if is_valid else Confidence.LOW
        conclusion = self.signal_evaluation.get("signal_conclusion", "Unformatted signal report.")
        
        return AgentResult(
            success=is_valid,
            data={"signal_evaluation": self.signal_evaluation},
            confidence=conf,
            evidence=[Evidence(source="signal_engine", content_snippet=f"{conclusion[:100]}...")],
            message=f"Biomedical signal analysis complete: {conclusion}"
        )
