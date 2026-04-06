import logging
import json
from typing import Dict, Any, List

from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from bio_ml_agent.llm_backend import auto_create_backend

log = logging.getLogger("structural.alphafold")


class AlphaFoldOrchestratorAgent(BaseSubAgent):
    """
    A1: AlphaFold Orchestrator Agent
    Takes a protein sequence or target identifier and orchestrates the structural prediction workflow.
    Decides between Local AlphaFold, ColabFold, or existing structure querying.
    Generates required artifacts like input sequences and configurations.
    """

    def __init__(self, model_name: str = "gemini-2.5-pro"):
        super().__init__("AlphaFoldOrchestratorAgent", model_name)
        self.llm = auto_create_backend(model_name)
        self.raw_input: str = ""
        self.job_configuration: Dict[str, Any] = {}

    def perceive(self, context: Dict[str, Any]) -> None:
        """Receives a raw protein sequence or target ID."""
        self.raw_input = context.get("sequence_or_target", "").strip()
        if not self.raw_input:
            log.warning("No sequence or target provided for AlphaFold orchestration.")
        else:
            log.info(f"🧬 AlphaFoldOrchestrator received input of length: {len(self.raw_input)}")

    def plan(self, goal: str) -> List[str]:
        return [
            "Validate input sequence/target format.",
            "Determine the optimal structural prediction pipeline (AF2/AF3, ColabFold, PDB query).",
            "Assess the necessity of MSA (Multiple Sequence Alignment) generation.",
            "Generate structured job artifacts (structure_job.json, run_config.yaml)."
        ]

    def act(self, step: str) -> Any:
        prompt = f"""
        You are the Master AlphaFold Orchestrator. The user has provided the following protein structural target (sequence or identifier):
        "{self.raw_input}"

        Your tasks:
        1. Validate if this is a FASTA sequence or a UniProt/PDB ID.
        2. Choose the pipeline: "local_alphafold", "colabfold_api", or "pdb_query".
        3. Determine if an MSA step is required prior to folding.
        4. Produce the job configuration.

        Respond STRICTLY with a JSON dictionary matching this schema:
        {{
            "is_valid_sequence": boolean,
            "detected_type": "fasta" | "id",
            "recommended_pipeline": "string",
            "msa_required": boolean,
            "run_config_yaml_content": "yaml string describing the run parameters",
            "structure_job_json": {{
                "target_name": "string",
                "length": int,
                "gpu_requirements": "string"
            }}
        }}
        Do NOT wrap in markdown unless it's just ```json.
        """
        messages = [{"role": "user", "content": prompt}]
        try:
            response = self.llm.chat(messages)
            clean_text = response.replace("```json", "").replace("```", "").strip()
            self.job_configuration = json.loads(clean_text)
            log.info("Successfully generated AlphaFold prediction configuration.")
        except Exception as e:
            log.error(f"Failed to parse orchestrator output: {e}")
            self.job_configuration["raw_response"] = response

        return f"Completed orchestration for AlphaFold."

    def verify(self, action_result: Any) -> bool:
        return "structure_job_json" in self.job_configuration

    def summarize(self) -> AgentResult:
        is_valid = self.verify("")
        conf = Confidence.HIGH if is_valid else Confidence.LOW

        return AgentResult(
            success=is_valid,
            data={"alphafold_job_config": self.job_configuration},
            confidence=conf,
            evidence=[Evidence(source="AlphaFoldOrchestrator", content_snippet="Run config generated.")],
            message=f"AlphaFold orchestration complete. Pipeline: {self.job_configuration.get('recommended_pipeline', 'Unknown')}"
        )
