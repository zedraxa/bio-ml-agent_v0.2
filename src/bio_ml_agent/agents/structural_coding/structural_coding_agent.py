import logging
import json
from typing import Dict, Any, List

from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from bio_ml_agent.llm_backend import auto_create_backend

log = logging.getLogger("structural_coding.core")

class StructuralCodingAgent(BaseSubAgent):
    """
    E1: Structural Coding Agent
    A highly specialized code generation unit tailored strictly to structural bioinformatics.
    Writes Python/Biopython scripts for PDB/mmCIF parsing, generic feature extraction, 
    and contact map generation.
    """

    def __init__(self, model_name: str = "gemini-2.5-pro"):
        super().__init__("StructuralCodingAgent", model_name)
        self.llm = auto_create_backend(model_name)
        self.script_request: Dict[str, Any] = {}
        self.generated_code: Dict[str, Any] = {}

    def perceive(self, context: Dict[str, Any]) -> None:
        """Receives prompt/request specific to parsing or measuring structural data."""
        self.script_request = context.get("coding_requirements", {})
        if not self.script_request:
            log.warning("No structural coding spec requested.")
        else:
            log.info("💻 StructuralCodingAgent received task parameters.")

    def plan(self, goal: str) -> List[str]:
        return [
            "Interpret request (e.g. 'Extract interface residues between Chain A and B').",
            "Determine optimal library (Bio.PDB, MDAnalysis, ProDy).",
            "Write highly robust, error-handled Python code block.",
            "Draft usage instructions for execution."
        ]

    def act(self, step: str) -> Any:
        if "refine" in step.lower() or "revision" in step.lower():
            # A4: Code Review Refinement
            self.refine_code(self.context.get("refinement_task", {}))
            return "Code refinement complete."

        prompt = f"""
        You are a Senior Structural Bioinformatics Software Engineer.
        Write a Python script adhering strictly to the following requirements:
        {json.dumps(self.script_request, indent=2)}

        Tasks:
        1. Write the Python code to perform robust PDB/mmCIF parsing or feature extraction (like contact maps).
        2. Keep dependencies minimal but standard (e.g. BioPython, numpy).
        3. Explain expected behavior and how to run it.

        Respond STRICTLY with a JSON dictionary matching:
        {{
            "filename": "extract_contact_map.py",
            "python_code": "import Bio ...",
            "required_packages": ["biopython", "numpy"],
            "execution_notes": "Run via `python extract_contact_map.py input.pdb`"
        }}
        """
        try:
            response = self.llm.chat([{"role": "user", "content": prompt}])
            clean_text = response.replace("```json", "").replace("```", "").strip()
            self.generated_code = json.loads(clean_text)
            log.info("Structural biopython code synthesized.")
        except Exception as e:
            log.error(f"Structural code generation failed: {e}")
            self.generated_code["raw_error_text"] = response

        return "Structural biology script developed."

    def refine_code(self, refinement_task: Dict[str, Any]):
        """
        A4: Code Review Refinement logic.
        Applies specific reviewer feedback to the generated code.
        """
        instruction = refinement_task.get("instruction", "")
        code_context = self.generated_code.get("python_code", "")
        
        log.info(f"💻 A4: Refining code based on review: '{instruction}'")
        
        prompt = f"""
        You are a Senior Structural Bioinformatics Software Engineer.
        A reviewer has provided the following feedback on your code:
        
        Feedback: {instruction}
        
        Original Code:
        {code_context}
        
        Tasks:
        1. Apply the feedback precisely (e.g., add type hints, refactor, fix paths).
        2. Ensure the code remains robust and functionally identical aside from the requested changes.
        
        Respond ONLY with the updated JSON dictionary (filename, python_code, etc.).
        """
        try:
            response = self.llm.chat([{"role": "user", "content": prompt}])
            clean_text = response.replace("```json", "").replace("```", "").strip()
            self.generated_code = json.loads(clean_text)
            log.info("Code refined based on review comments.")
        except Exception as e:
            log.error(f"Code refinement failed: {e}")

    def verify(self, action_result: Any) -> bool:
        return "python_code" in self.generated_code

    def summarize(self) -> AgentResult:
        is_valid = self.verify("")
        conf = Confidence.HIGH if is_valid else Confidence.LOW
        return AgentResult(
            success=is_valid,
            data={"structural_script": self.generated_code},
            confidence=conf,
            evidence=[Evidence("StructuralCodingAgent", "Drafted custom BioPython utility")],
            message="Structural helper code created."
        )
