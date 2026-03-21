import logging
from typing import Dict, Any, List
import json

from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from bio_ml_agent.llm_backend import auto_create_backend

log = logging.getLogger("biology.biostats")

class BiostatisticsAgent(BaseSubAgent):
    """
    C7: Biostatistics Agent
    Bilimsel verinin güvenirliğini ölçen metodoloji uzmanıdır. Hangi istatistiksel testin
    uygulanması gerektiğine otonom karar verir, yanlış yorum denetimi yapar.
    
    Yetenekleri:
    - Uygun test önerileri (ANOVA, Kruskal-Wallis, t-test vb.)
    - Normality (Shapiro-Wilk vb.) / Variance (Levene) homojenlik mantığı kurma
    - Sample structure (boyut, eşleştirilme durumu) analizi
    - Effect size (Etki büyüklüğü: Cohen's d vs) notları
    - Yanlış yorum riskleri (P-hacking vb.)
    - Grafik ve tablo formatı üretim tavsiyeleri
    """

    def __init__(self, model_name: str = "gemini-2.5-pro"):
        super().__init__("BiostatisticsAgent", model_name)
        self.llm = auto_create_backend(model_name)
        self.experimental_schema: Dict[str, Any] = {}
        self.stat_evaluation: Dict[str, Any] = {}

    def perceive(self, context: Dict[str, Any]) -> None:
        """Verinin şemasını (dağılım özellikleri, sample N sayısı) alır."""
        self.experimental_schema = context.get("experimental_schema", {})
        if not self.experimental_schema:
            log.warning("No design/schema data provided to BiostatisticsAgent.")
        else:
            log.info(f"📊 BiostatsAgent received schema features.")

    def plan(self, goal: str) -> List[str]:
        return [
            "Perform comprehensive Biostatistical methodology design (Tests, Assumptions, Effect Sizes, Graphics)."
        ]

    def act(self, step: str) -> Any:
        prompt = f"""
        You are a highly pedantic Senior Biostatistician methodology reviewer.
        Analyze the following experimental design schema / sample features.
        
        Experimental Data Schema:
        {json.dumps(self.experimental_schema, indent=2)}
        
        Provide the strict statistical road-map required to test hypotheses on this data.
        Respond STRICTLY with a JSON dictionary containing your assessment:
        {{
            "appropriate_test_suggestions": "t-test/ANOVA/non-parametric along with exact reasons...",
            "normality_and_variance_assumptions": "tests to apply before the main test...",
            "sample_structure_and_power": "interpretation on N size limitations...",
            "effect_size_metrics": "which metrics to report (e.g. Cohen's d, Eta squared)...",
            "misinterpretation_risks": "warning against p-hacking or false correlations...",
            "graphic_and_table_generation_directives": "which plots strictly correspond to this distribution..."
        }}
        Do NOT write navigational text. Only return the JSON (optionally wrapped in ```json markdown).
        """
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response_text = self.llm.chat(messages)
            clean_text = response_text.replace("```json", "").replace("```", "").strip()
            self.stat_evaluation = json.loads(clean_text)
        except Exception as e:
            log.warning(f"Biostats parsing fell back to strings. Error: {e}")
            self.stat_evaluation["unformatted_evaluation"] = response_text
            
        return f"Completed Biostatistics Evaluation"

    def verify(self, action_result: Any) -> bool:
        keys = ["appropriate_test_suggestions", "misinterpretation_risks"]
        return any(k in self.stat_evaluation for k in keys)

    def summarize(self) -> AgentResult:
        is_valid = self.verify("")
        conf = Confidence.HIGH if is_valid else Confidence.LOW
        
        return AgentResult(
            success=is_valid,
            data={"biostats_protocol": self.stat_evaluation},
            confidence=conf,
            evidence=[Evidence(source="biostats_engine", content_snippet="Stats setup staged.")],
            message=f"Biostatistical methodology parsing complete."
        )
