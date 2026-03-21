import logging
import json
from typing import Dict, Any, List

from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from bio_ml_agent.llm_backend import auto_create_backend

log = logging.getLogger("omics.pathway_reasoning")

class PathwayReasoningAgent(BaseSubAgent):
    """
    D2: Pathway Reasoning Agent
    Maps prioritized gene/protein lists to actual biological processes.
    Builds mechanistic summaries and traces upstream/downstream regulatory networks.
    """

    def __init__(self, model_name: str = "gemini-2.5-pro"):
        super().__init__("PathwayReasoningAgent", model_name)
        self.llm = auto_create_backend(model_name)
        self.pathway_context: Dict[str, Any] = {}
        self.mechanistic_summary: Dict[str, Any] = {}

    def perceive(self, context: Dict[str, Any]) -> None:
        """Receives a list of genes/proteins from expression agents or structural hits."""
        self.pathway_context = context.get("gene_set", {})
        if not self.pathway_context:
            log.warning("No gene set provided for pathway mapping.")
        else:
            log.info("🌐 PathwayReasoningAgent received gene cluster array.")

    def plan(self, goal: str) -> List[str]:
        return [
            "Perform conceptual Over-Representation Analysis (ORA) via LLM pathways DB.",
            "Group candidate genes into functional mechanistic clusters.",
            "Deduce upstream regulatory triggers causing this phenotype.",
            "Identify downstream metabolic/signaling consequences."
        ]

    def act(self, step: str) -> Any:
        prompt = f"""
        You are an expert Systems Biologist. 
        Analyze the following cluster of differentially expressed genes or structural hits:
        {json.dumps(self.pathway_context, indent=2)}

        Tasks:
        1. Identify the primary KEGG/Reactome biological pathways dominating this set.
        2. Group the genes into specific Functional Mechanisms (e.g. 'Apoptotic Regulation').
        3. Formulate hypotheses on the upstream regulators (e.g. driven by p53 loss).
        4. Detail downstream organismal or cellular consequences.

        Respond STRICTLY with a JSON dictionary matching:
        {{
            "dominant_pathways": ["PI3K-Akt signaling", "Cell cycle arrest"],
            "mechanistic_summary_md": "Markdown text explaining how the genes work together to force cell cycle arrest",
            "upstream_regulators_hypothesis": ["MYC hyperactivation", "Hypoxia"],
            "downstream_consequences": ["Inhibition of apoptosis", "Increased glycolysis"]
        }}
        """
        try:
            response = self.llm.chat([{"role": "user", "content": prompt}])
            clean_text = response.replace("```json", "").replace("```", "").strip()
            self.mechanistic_summary = json.loads(clean_text)
            log.info("Pathway and mechanistic reasoning logic derived.")
        except Exception as e:
            log.error(f"Pathway reasoning parsing failed: {e}")
            self.mechanistic_summary["raw_error_text"] = response

        return "Mechanistic pathway network extracted."

    def verify(self, action_result: Any) -> bool:
        return "mechanistic_summary_md" in self.mechanistic_summary

    def summarize(self) -> AgentResult:
        is_valid = self.verify("")
        conf = Confidence.HIGH if is_valid else Confidence.LOW
        return AgentResult(
            success=is_valid,
            data={"pathway_reasoning": self.mechanistic_summary},
            confidence=conf,
            evidence=[Evidence("PathwayReasoningAgent", "Built mechanistic map from isolated genes")],
            message="Gene clusters successfully connected to biological pathways."
        )
