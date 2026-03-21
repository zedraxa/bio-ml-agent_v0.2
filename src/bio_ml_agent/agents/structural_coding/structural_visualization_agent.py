import logging
import json
from typing import Dict, Any, List

from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from bio_ml_agent.llm_backend import auto_create_backend

log = logging.getLogger("structural_coding.visualization")

class StructuralVisualizationAgent(BaseSubAgent):
    """
    E4: Structural Visualization Agent
    Automates the creation of high-quality, report-ready protein imagery.
    Produces scripts for PyMOL or UCSF ChimeraX to color structures by pLDDT,
    highlight domains, and isolate ligand binding interactions.
    """

    def __init__(self, model_name: str = "gemini-2.5-pro"):
        super().__init__("StructuralVisualizationAgent", model_name)
        self.llm = auto_create_backend(model_name)
        self.viz_specs: Dict[str, Any] = {}
        self.generated_scripts: Dict[str, Any] = {}

    def perceive(self, context: Dict[str, Any]) -> None:
        """Receives visualization targets (e.g. highlight residues 50-70 in red)."""
        self.viz_specs = context.get("visualization_targets", {})
        if not self.viz_specs:
            log.warning("No visualization targets provided.")
        else:
            log.info("🎨 StructuralVisualizationAgent received graphic specifications.")

    def plan(self, goal: str) -> List[str]:
        return [
            "Interpret required visualization task (e.g. pLDDT spectrum).",
            "Select appropriate software target (PyMOL `.pml` or ChimeraX `.cxc`).",
            "Generate raw executable macro code for high-res ray-traced rendering.",
            "Formulate export and UI helper instructions."
        ]

    def act(self, step: str) -> Any:
        prompt = f"""
        You are an expert Molecular Illustrator and PyMOL/ChimeraX Scripter.
        Generate rendering scripts based on these graphic specifications:
        {json.dumps(self.viz_specs, indent=2)}

        Tasks:
        1. If highlighting confidence, generate a PyMOL script that colors b-factors (pLDDT) from red (low) to blue (high).
        2. If highlighting a ligand pocket, generate scripts to show surface meshes around the ligand.
        3. Include commands for ray tracing (high resolution PNG export).

        Respond STRICTLY with a JSON dictionary matching:
        {{
            "software_target": "PyMOL",
            "script_filename": "render_plddt.pml",
            "script_content": "load target.pdb\\ncolor b\\nray 1200,1200\\npng output.png",
            "instructions": "Run this script using `pymol -cq render_plddt.pml` for headless rendering."
        }}
        """
        try:
            response = self.llm.chat([{"role": "user", "content": prompt}])
            clean_text = response.replace("```json", "").replace("```", "").strip()
            self.generated_scripts = json.loads(clean_text)
            log.info("Molecular rendering scripts created.")
        except Exception as e:
            log.error(f"Visualization script generation failed: {e}")
            self.generated_scripts["raw_error_text"] = response

        return "Report-ready visualizations modeled."

    def verify(self, action_result: Any) -> bool:
        return "script_content" in self.generated_scripts

    def summarize(self) -> AgentResult:
        is_valid = self.verify("")
        conf = Confidence.HIGH if is_valid else Confidence.LOW
        return AgentResult(
            success=is_valid,
            data={"visualization_macros": self.generated_scripts},
            confidence=conf,
            evidence=[Evidence("StructuralVisualization", "Generated PyMOL/ChimeraX macros")],
            message="Graphical molecular macros built."
        )
