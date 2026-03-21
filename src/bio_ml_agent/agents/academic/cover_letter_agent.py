import logging
from typing import Dict, Any
from bio_ml_agent.core.agent_base import BaseSubAgent

logger = logging.getLogger("academic.cover_letter_agent")

class CoverLetterAgent(BaseSubAgent):
    """
    F4: Cover Letter Agent
    Drafts the formal submission cover letter to the journal Editor.
    """
    def __init__(self):
        super().__init__(
            role_name="CoverLetterWriter",
            system_prompt=(
                "You are an Academic Submission Strategist. You write the Cover Letter "
                "to the Editor-in-Chief. You must assert why the manuscript is a perfect fit "
                "for the specific journal, state that it is original work, and warmly summarize the impact. Output JSON."
            )
        )
        self.manuscript_summary = ""

    def perceive(self, context: Dict[str, Any]):
        self.manuscript_summary = context.get("manuscript_summary", "")
        self.journal_name = context.get("journal_name", "Unknown Journal")
        self.editor_name = context.get("editor_name", "Editor-in-Chief")

    def plan(self) -> str:
        return f"Draft submission cover letter to {self.journal_name}."

    def act(self, instructions: str) -> None:
        prompt = f"""
        Journal: {self.journal_name}
        Editor: {self.editor_name}
        Manuscript Summary:
        {self.manuscript_summary}
        
        Format as JSON:
        {{
            "cover_letter_md": "Formal business letter format to the journal",
            "suggested_reviewers": ["Dummy names/profiles if applicable"]
        }}
        """
        self.current_result = self.llm.chat([{"role": "user", "content": prompt}])

    def verify(self) -> bool:
        return "cover_letter_md" in self.current_result

    def summarize(self) -> Any:
        try:
            import json
            data = json.loads(self.current_result.replace("```json", "").replace("```", "").strip())
        except Exception:
            data = {"raw": self.current_result}
        return self.create_checkpoint("Write Cover Letter", data, "HIGH")
