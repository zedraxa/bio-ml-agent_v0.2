import logging
import json
from typing import Dict, Any, List

from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from bio_ml_agent.llm_backend import auto_create_backend

log = logging.getLogger("structural.virtual_screening")

class VirtualScreeningOrchestrator(BaseSubAgent):
    """
    C2: Virtual Screening Orchestrator
    Scales molecular docking into High-Throughput Virtual Screening (HTVS).
    Handles library filtering, job batching, and scoring aggregation.
    """

    def __init__(self, model_name: str = "gemini-2.5-pro"):
        super().__init__("VirtualScreeningOrchestrator", model_name)
        self.llm = auto_create_backend(model_name)
        self.screening_context: Dict[str, Any] = {}
        self.screening_job: Dict[str, Any] = {}

    def perceive(self, context: Dict[str, Any]) -> None:
        """Receives chemical library size, computational constraints, and target setup."""
        self.screening_context = context.get("screening_parameters", {})
        if not self.screening_context:
            log.warning("No screening parameters provided.")
        else:
            log.info("💻 VirtualScreeningOrchestrator received screening constraints.")

    def plan(self, goal: str) -> List[str]:
        return [
            "Evaluate target protein constraints.",
            "Determine library filtering rules (e.g. Lipinski's Rule of 5).",
            "Establish job batching architecture for distributed environments.",
            "Generate durable workflow queue setup for scoring aggregation."
        ]

    def act(self, step: str) -> Any:
        prompt = f"""
        You are the Master Virtual Screening Orchestrator. 
        Analyze the constraints for the upcoming screening campaign:
        {json.dumps(self.screening_context, indent=2)}

        Tasks:
        1. Select specific chemical library filters (e.g. MW limits, LogP limits).
        2. Divide the screening into manageable batch runs.
        3. Define the aggregation strategy for harvesting docking scores into a CSV.

        Respond STRICTLY with a JSON dictionary matching:
        {{
            "library_filters": ["MW < 500", "H-bond donors <= 5"],
            "batch_strategy": "Divide 1M compounds into 10K batches across clusters.",
            "score_aggregation_logic": "Select top pose per ligand, sort by binding affinity",
            "screening_summary_md": "Markdown campaign summary text"
        }}
        """
        try:
            response = self.llm.chat([{"role": "user", "content": prompt}])
            clean_text = response.replace("```json", "").replace("```", "").strip()
            self.screening_job = json.loads(clean_text)
            log.info("HTVS batching and filtering initialized.")
        except Exception as e:
            log.error(f"Orchestrator parsing failed: {e}")
            self.screening_job["raw_error_text"] = response

        return "Virtual screening scale-up logic planned."

    def verify(self, action_result: Any) -> bool:
        return "batch_strategy" in self.screening_job

    def summarize(self) -> AgentResult:
        is_valid = self.verify("")
        conf = Confidence.HIGH if is_valid else Confidence.LOW
        return AgentResult(
            success=is_valid,
            data={"virtual_screening_setup": self.screening_job},
            confidence=conf,
            evidence=[Evidence("VirtualScreeningOrchestrator", "Calculated distributed HTVS queueing")],
            message="High-Throughput Virtual Screening orchestrated."
        )
