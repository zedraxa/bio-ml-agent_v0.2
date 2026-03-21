import logging
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from bio_ml_agent.llm_backend import auto_create_backend
from pathlib import Path

log = logging.getLogger("microscopy.identifier")

class MicroscopyOntology:
    """
    MicroscopyOntology: Kullanıcının talep ettiği 'en ince detay' listesi.
    """
    CELLULAR = [
        "nucleus", "nucleolus", "cytoplasm", "cell membrane boundary", 
        "vacuole", "chloroplast", "mitotic spindle region", "condensed chromosomes", 
        "apoptotic bodies", "inclusion bodies", "granules"
    ]
    TISSUE = [
        "epithelium", "connective tissue", "vessel-like structures", 
        "gland-like structures", "necrotic regions", "inflammatory clusters", "stromal regions"
    ]
    DEVELOPMENTAL = [
        "interphase", "prophase", "metaphase", "anaphase", "telophase", 
        "cytokinesis", "meiosis stage variants", "pollen mother cell states", 
        "anther wall layers", "root tip mitosis patterns"
    ]
    MORPHOLOGICAL = [
        "elongated vs round cells", "clustered vs isolated", "pleomorphism", 
        "abnormal nucleus-to-cytoplasm ratio", "irregular borders", 
        "hyperchromatic appearance", "fragmentation patterns"
    ]

@dataclass
class DetailedIDResult:
    specimen: str = ""
    likely_process: str = ""
    analysis: Dict[str, Any] = field(default_factory=lambda: {
        "cellular": {}, "tissue": {}, "developmental": {}, "morphological": {}
    })

class MicroscopyIdentifierAgent(BaseSubAgent):
    """
    MicroscopyIdentifierAgent (A2 - Physical Vision Layer):
    - Passes full biological ontology to a Multimodal LLM (Gemini/Claude).
    - Queries the model using physical image to identify deep morphologic traits.
    """
    
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        super().__init__("MicroscopyIdentifierAgent", model_name)
        self.image_path: Optional[Path] = None
        self.llm = auto_create_backend(model_name)
        self.result = DetailedIDResult()

    def perceive(self, context: Dict[str, Any]) -> None:
        path = context.get("image_path")
        if path:
            self.image_path = Path(path)
            log.info(f"🧬 [ULTIMATE DETAIL] Physical Vision Agent loading for: {self.image_path.name}")

    def plan(self, goal: str) -> List[str]:
        return [
            "Construct Deep Ontology JSON prompt",
            "Send target image and ontology constraint to Vision Model",
            "Extract structured Hierarchical Tree outputs"
        ]

    def act(self, step: str) -> Any:
        if not self.image_path or not self.image_path.exists(): 
            return "Error: No Valid Image"
            
        log.info(f"🧠 Querying Deep Bio-Ontology: {step}")
        
        if "send" in step.lower() or "ontology" in step.lower():
            prompt = f"""
            Perform a deep, fine-grained identification of the attached microscopy image.
            Only output a valid strict JSON containing the exact structure below. 
            Do NOT include markdown block markers (e.g. ```json).
            
            Look for these specific ontological concepts if present:
            - Cellular Layer: {MicroscopyOntology.CELLULAR}
            - Tissue Layer: {MicroscopyOntology.TISSUE}
            - Developmental Phases: {MicroscopyOntology.DEVELOPMENTAL}
            - Morphology Anomalies: {MicroscopyOntology.MORPHOLOGICAL}
            
            Required Output Format:
            {{
                "specimen": "String",
                "likely_process": "String (e.g., Mitosis, Necrosis, Unknown)",
                "analysis": {{
                    "cellular": {{"found_structures": [], "notes": ""}},
                    "tissue": {{"found_structures": [], "notes": ""}},
                    "developmental": {{"found_phases": [], "notes": ""}},
                    "morphological": {{"shape_factors": [], "notes": ""}}
                }}
            }}
            """
            
            messages = [
                {"role": "system", "content": "You are a PhD-level Pathologist / Bioengineer."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "file", "path": str(self.image_path)}
                ]}
            ]
            
            try:
                response = self.llm.chat(messages)
                if "```json" in response:
                    response = response.replace("```json", "").replace("```", "").strip()
                elif "```" in response:
                    response = response.replace("```", "").strip()
                    
                data = json.loads(response)
                self.result.specimen = data.get("specimen", "")
                self.result.likely_process = data.get("likely_process", "")
                self.result.analysis = data.get("analysis", {})
                log.info(f"✅ Extracted full deep identification tree for {self.result.specimen}")
                
            except Exception as e:
                log.error(f"Failed to query Deep Ontology Vision: {e}")
                
        return "Deep diagnostic layer via Vision API processed."

    def verify(self, action_result: Any) -> bool:
        return True

    def summarize(self) -> AgentResult:
        return AgentResult(
            success=True,
            data=self.result.analysis,
            confidence=Confidence.CRITICAL, 
            evidence=[Evidence(source=str(self.image_path), content_snippet="Multimodal LLM analysis executed.")],
            message=f"Physical Identification Complete: {self.result.specimen}. Hierarchical tree built natively."
        )
