import logging
from typing import Dict, Any, List
import json

from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from bio_ml_agent.llm_backend import auto_create_backend

log = logging.getLogger("biology.mbg")

class MBGAgent(BaseSubAgent):
    """
    C3: Molecular Biology / Genetics (MBG) Agent
    Biyoinformatik kodlayıcı ajanın çıktı dizilerini (sequence, expression matrix vb.)
    moleküler biyoloji ve genetik perspektifinden anlamlandıran validasyon ajanıdır.
    
    Yetenekleri:
    - DNA/RNA/Protein İş Akışı Yorumlaması
    - Primer Mantığı (Thermodynamics, off-target yorumu)
    - Sequence & Annotation Yorumları
    - Genotype / Phenotype Notları ve Hastalık İlişkisi
    - Pathway (Yolak) / Gene Set Özetleme
    - Expression (RNA-Seq/Microarray) Data Yorumlama
    """

    def __init__(self, model_name: str = "gemini-2.5-pro"):
        super().__init__("MBGAgent", model_name)
        self.llm = auto_create_backend(model_name)
        self.molecular_data: Dict[str, Any] = {}
        self.molecular_evaluation: Dict[str, Any] = {}

    def perceive(self, context: Dict[str, Any]) -> None:
        """Dizi analiz metrikleri, gen ifadeleri, varyant (VCF) notları veya primer dizilerini alır."""
        self.molecular_data = context.get("molecular_data", {})
        if not self.molecular_data:
            log.warning("No molecular data provided to MBGAgent.")
        else:
            log.info(f"🧬 MBGAgent received datasets: {list(self.molecular_data.keys())}")

    def plan(self, goal: str) -> List[str]:
        return [
            "Perform comprehensive Molecular Biology and Genetic evaluation (Expression, Pathway, Phenotype, Sequenece)."
        ]

    def act(self, step: str) -> Any:
        prompt = f"""
        You are a Senior Molecular Biologist and Geneticist.
        Analyze the following omics/sequence data structures.
        
        Molecular Data:
        {json.dumps(self.molecular_data, indent=2)}
        
        Provide a deep biological/genetic interpretation.
        Respond STRICTLY with a JSON dictionary containing your assessment:
        {{
            "sequence_and_annotation_insights": "interpretation...",
            "primer_and_workflow_logic": "interpretation...",
            "genotype_to_phenotype_notes": "interpretation...",
            "pathway_and_gene_set_summary": "interpretation...",
            "expression_data_interpretation": "interpretation...",
            "molecular_conclusion": "overall genetic / mechanistic summary..."
        }}
        Do NOT write navigational text. Only return the JSON (optionally wrapped in ```json markdown).
        """
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response_text = self.llm.chat(messages)
            clean_text = response_text.replace("```json", "").replace("```", "").strip()
            self.molecular_evaluation = json.loads(clean_text)
        except Exception as e:
            log.warning(f"MBG parsing fell back to strings. Error: {e}")
            self.molecular_evaluation["unformatted_evaluation"] = response_text
            
        return f"Completed Molecular Biology Evaluation"

    def verify(self, action_result: Any) -> bool:
        keys = ["pathway_and_gene_set_summary", "molecular_conclusion"]
        return any(k in self.molecular_evaluation for k in keys)

    def summarize(self) -> AgentResult:
        is_valid = self.verify("")
        conf = Confidence.HIGH if is_valid else Confidence.LOW
        conclusion = self.molecular_evaluation.get("molecular_conclusion", "Unformatted molecular report.")
        
        return AgentResult(
            success=is_valid,
            data={"molecular_evaluation": self.molecular_evaluation},
            confidence=conf,
            evidence=[Evidence(source="mbg_engine", content_snippet=f"{conclusion[:100]}...")],
            message=f"Molecular analysis complete: {conclusion}"
        )
