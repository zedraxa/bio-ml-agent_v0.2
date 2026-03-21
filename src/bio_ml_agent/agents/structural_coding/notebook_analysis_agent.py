import logging
import json
from typing import Dict, Any, List

from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from bio_ml_agent.llm_backend import auto_create_backend

log = logging.getLogger("structural_coding.notebook")

class NotebookToAnalysisAgent(BaseSubAgent):
    """
    E3: Notebook-to-Analysis Agent
    Generates interactive Jupyter/Colab notebooks (.ipynb) dynamically for structural 
    exploration, confidence plotting, and docking data analysis.
    """

    def __init__(self, model_name: str = "gemini-2.5-pro"):
        super().__init__("NotebookToAnalysisAgent", model_name)
        self.llm = auto_create_backend(model_name)
        self.notebook_specs: Dict[str, Any] = {}
        self.generated_notebook: Dict[str, Any] = {}

    def perceive(self, context: Dict[str, Any]) -> None:
        """Receives specifications for what analytical views the notebook should contain."""
        self.notebook_specs = context.get("notebook_requirements", {})
        if not self.notebook_specs:
            log.warning("No notebook specs received.")
        else:
            log.info("📓 NotebookToAnalysisAgent received plot and data specs.")

    def plan(self, goal: str) -> List[str]:
        return [
            "Parse requested plotting inputs (e.g. pLDDT arrays, RMSD series).",
            "Generate raw JSON mapping to a valid `.ipynb` file specification.",
            "Include markdown cells for report narration.",
            "Include Python cells with matplotlib/seaborn code for visual rendering."
        ]

    def act(self, step: str) -> Any:
        prompt = f"""
        You are an elite Data Scientist specializing in Bioinformatics Notebooks.
        Generate the blueprint for a Jupyter Notebook based on these specs:
        {json.dumps(self.notebook_specs, indent=2)}

        Tasks:
        1. Emulate the structure of an .ipynb file containing Markdown and Code cells.
        2. Write matplotlib/seaborn code to graph the required data (e.g. pLDDT distribution).
        3. Insert explanatory markdown narrative.

        Respond STRICTLY with a JSON dictionary matching:
        {{
            "notebook_filename": "structural_analysis.ipynb",
            "cells": [
                {{"type": "markdown", "source": "# Protein Structure Confidence Analysis\\n"}},
                {{"type": "code", "source": "import matplotlib.pyplot as plt\\nimport seaborn as sns\\n"}}
            ],
            "execution_notes": "Use JupyterLab to run and view interactive plots."
        }}
        """
        try:
            response = self.llm.chat([{"role": "user", "content": prompt}])
            clean_text = response.replace("```json", "").replace("```", "").strip()
            self.generated_notebook = json.loads(clean_text)
            log.info("Interactive Notebook structure synthesized.")
        except Exception as e:
            log.error(f"Notebook generation failed: {e}")
            self.generated_notebook["raw_error_text"] = response

        return "Data science notebook templates finalized."

    def verify(self, action_result: Any) -> bool:
        return "cells" in self.generated_notebook

    def summarize(self) -> AgentResult:
        is_valid = self.verify("")
        conf = Confidence.HIGH if is_valid else Confidence.LOW
        return AgentResult(
            success=is_valid,
            data={"jupyter_notebook_struct": self.generated_notebook},
            confidence=conf,
            evidence=[Evidence("NotebookAgent", "Created interactive plotting pipeline")],
            message="Analytical Jupyter Notebook generated."
        )
