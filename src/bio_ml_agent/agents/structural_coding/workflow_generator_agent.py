import logging
import json
from typing import Dict, Any, List

from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from bio_ml_agent.llm_backend import auto_create_backend

log = logging.getLogger("structural_coding.workflow")

class WorkflowGeneratorAgent(BaseSubAgent):
    """
    E2: Workflow Generator Agent
    Specializes in IT/HPC orchestration for structural modeling.
    Outputs bash/shell wrappers for AlphaFold pipelines, HTVS docking arrays, 
    and Slurm configuration scripts.
    """

    def __init__(self, model_name: str = "gemini-2.5-pro"):
        super().__init__("WorkflowGeneratorAgent", model_name)
        self.llm = auto_create_backend(model_name)
        self.pipeline_specs: Dict[str, Any] = {}
        self.workflow_scripts: Dict[str, Any] = {}

    def perceive(self, context: Dict[str, Any]) -> None:
        """Receives compute environment configs and desired modeling workflow specs."""
        self.pipeline_specs = context.get("hpc_workflow_requirements", {})
        if not self.pipeline_specs:
            log.warning("No workflow specs provided.")
        else:
            log.info("⚙️ WorkflowGeneratorAgent received pipeline config.")

    def plan(self, goal: str) -> List[str]:
        return [
            "Parse requested pipeline topology (e.g. Local AlphaFold + AutoDock Vina).",
            "Generate HPC SLURM submission scripts based on environment specs.",
            "Write bash wrapper scripts to iterate over batches.",
            "Provide file-manifest outputs (run configurations)."
        ]

    def act(self, step: str) -> Any:
        prompt = f"""
        You are a High-Performance Computing (HPC) architect for Bioinformatics.
        Design a workflow pipeline utilizing these specs:
        {json.dumps(self.pipeline_specs, indent=2)}

        Tasks:
        1. Generate a main executable shell wrapper (e.g., looping over fasta files for local AlphaFold).
        2. Generate an optional SLURM or PBS batch submission script indicating GPU usage.
        3. Draft standard YAML configurations if a framework (like Snakemake or Nextflow) is implied.

        Respond STRICTLY with a JSON dictionary matching:
        {{
            "bash_wrapper_script": "#!/bin/bash\\n...",
            "slurm_submission_script": "#!/bin/bash\\n#SBATCH --gpus=1\\n...",
            "run_manifest_yaml": "experiment: batch_docking\\n...",
            "deployment_instructions": "sbatch run_alphafold.sh"
        }}
        """
        try:
            response = self.llm.chat([{"role": "user", "content": prompt}])
            clean_text = response.replace("```json", "").replace("```", "").strip()
            self.workflow_scripts = json.loads(clean_text)
            log.info("Workflow pipeline configuration completed.")
        except Exception as e:
            log.error(f"Workflow generation parsing failed: {e}")
            self.workflow_scripts["raw_error_text"] = response

        return "Run configuration and wrapping completed."

    def verify(self, action_result: Any) -> bool:
        return "bash_wrapper_script" in self.workflow_scripts

    def summarize(self) -> AgentResult:
        is_valid = self.verify("")
        conf = Confidence.HIGH if is_valid else Confidence.LOW
        return AgentResult(
            success=is_valid,
            data={"workflow_hpc_scripts": self.workflow_scripts},
            confidence=conf,
            evidence=[Evidence("WorkflowGeneratorAgent", "Generated shell/SLURM scripts")],
            message="High-Performance Computing pipeline wrapped."
        )
