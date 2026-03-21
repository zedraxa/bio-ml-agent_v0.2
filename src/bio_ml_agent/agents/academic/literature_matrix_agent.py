import logging
from typing import Dict, Any, List
from bio_ml_agent.core.agent_base import BaseSubAgent

logger = logging.getLogger("academic.literature_matrix")

class LiteratureMatrixAgent(BaseSubAgent):
    """
    C1: Literature Matrix Agent
    Parses multiple papers and outputs a CSV/Markdown comparative matrix
    capturing methods, sample sizes, main findings, and limitations.
    """
    def __init__(self):
        super().__init__(
            role_name="LiteratureMatrix",
            system_prompt=(
                "You are an Academic Literature Synthesizer. "
                "Your job is to read summaries or abstracts of multiple papers "
                "and organize them into a structured comparative matrix (Author, Year, Method, Findings, Limits). "
                "Output JSON containing the markdown table."
            )
        )
        self.papers = []

    def perceive(self, context: Dict[str, Any]):
        self.papers = context.get("papers", [])

    def plan(self) -> str:
        return "Extract structured features from papers and format as a comparative matrix."

    def act(self, instructions: str) -> None:
        prompt = f"""
        Analyze the following literature bundle:
        {self.papers}
        
        Create a comparative Literature Matrix. Output JSON:
        {{
            "matrix_md": "Markdown table format mapping Author/Year, Method, Main Finding, Limitations",
            "matrix_csv": "CSV string format of the same data",
            "synthesis_summary": "1-2 sentences summarizing the overall state of the papers"
        }}
        """
        self.current_result = self.llm.chat([{"role": "user", "content": prompt}])

    def verify(self) -> bool:
        return "matrix_md" in self.current_result

    def summarize(self) -> Any:
        try:
            import json
            data = json.loads(self.current_result.replace("```json", "").replace("```", "").strip())
        except Exception:
            data = {"raw": self.current_result}
        return self.create_checkpoint("Build Literature Matrix", data, "HIGH")
