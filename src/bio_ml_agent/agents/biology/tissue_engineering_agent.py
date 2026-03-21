import logging
from typing import Dict, Any, List
import json

from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from bio_ml_agent.llm_backend import auto_create_backend

log = logging.getLogger("biology.tissue_engineering")

class TissueEngineeringAgent(BaseSubAgent):
    """
    C5: Tissue Engineering Agent
    Doku mühendisliği ve biyomateryal araştırmaları için son derece özgün 
    materyal-analiz ve deney planlama ajanıdır.
    
    Yetenekleri:
    - Scaffold Material Comparison (Biyomateryal iskeleti kıyaslaması)
    - Pore Structure Metrics (Porozite, yüzey alanı vb. metriklerin biyolojik yorumu)
    - Cell-Scaffold Interaction (Hücre-iskelet etkileşimi ve mekanobiyolojik yorum)
    - Biomaterial Property Tables (Karşılaştırmalı özellik tabloları çıkarma)
    - Experimental Planning (Hücre ekimi (seeding) ve kültivasyon deney kurgusu)
    """

    def __init__(self, model_name: str = "gemini-2.5-pro"):
        super().__init__("TissueEngineeringAgent", model_name)
        self.llm = auto_create_backend(model_name)
        self.material_data: Dict[str, Any] = {}
        self.tissue_evaluation: Dict[str, Any] = {}

    def perceive(self, context: Dict[str, Any]) -> None:
        """Mikroskobik görüntü (SEM vb.) metriklerini veya materyal test datalarını alır."""
        self.material_data = context.get("material_data", {})
        if not self.material_data:
            log.warning("No material/scaffold data provided to TissueEngineeringAgent.")
        else:
            log.info(f"🦴 TissueEngAgent received material metric keys: {list(self.material_data.keys())}")

    def plan(self, goal: str) -> List[str]:
        return [
            "Perform comprehensive Tissue Engineering & Biomaterials evaluation (Scaffolds, Cells, Planning)."
        ]

    def act(self, step: str) -> Any:
        prompt = f"""
        You are a highly specialized Tissue Engineering and Biomaterials Expert.
        Analyze the following experimental material properties, pore metrics, or cell-scaffold interaction data.
        
        Material / Cellular Data:
        {json.dumps(self.material_data, indent=2)}
        
        Provide a world-class biomaterials engineering interpretation.
        Respond STRICTLY with a JSON dictionary containing your assessment:
        {{
            "scaffold_material_comparison": "analytical interpretation...",
            "pore_structure_and_mechanics": "interpretation of pore size/distribution...",
            "cell_scaffold_interaction_synth": "literature-backed cell adhesion/proliferation interpretation...",
            "biomaterial_property_synthesis": "compiled insights for property tables...",
            "experimental_planning_recommendation": "step-by-step biological & material test planning...",
            "engineering_conclusion": "overall material viability summary..."
        }}
        Do NOT write navigational text. Only return the JSON (optionally wrapped in ```json markdown).
        """
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response_text = self.llm.chat(messages)
            clean_text = response_text.replace("```json", "").replace("```", "").strip()
            self.tissue_evaluation = json.loads(clean_text)
        except Exception as e:
            log.warning(f"TissueEng parsing fell back to strings. Error: {e}")
            self.tissue_evaluation["unformatted_evaluation"] = response_text
            
        return f"Completed Tissue Engineering Evaluation"

    def verify(self, action_result: Any) -> bool:
        keys = ["scaffold_material_comparison", "experimental_planning_recommendation"]
        return any(k in self.tissue_evaluation for k in keys)

    def summarize(self) -> AgentResult:
        is_valid = self.verify("")
        conf = Confidence.HIGH if is_valid else Confidence.LOW
        conclusion = self.tissue_evaluation.get("engineering_conclusion", "Unformatted bio-engineering report.")
        
        return AgentResult(
            success=is_valid,
            data={"tissue_engineering_evaluation": self.tissue_evaluation},
            confidence=conf,
            evidence=[Evidence(source="tissue_engine", content_snippet=f"{conclusion[:100]}...")],
            message=f"Tissue engineering analysis complete: {conclusion}"
        )
