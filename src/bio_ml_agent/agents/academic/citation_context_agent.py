import logging
from typing import Dict, Any, List
from bio_ml_agent.core.agent_base import BaseSubAgent

logger = logging.getLogger("academic.citation_context")

class CitationContextAgent(BaseSubAgent):
    """
    C2: Citation Context Agent
    Determines where and how to embed citations into an ongoing draft.
    Provides context (supportive vs contrary).
    """
    def __init__(self):
        super().__init__(
            role_name="CitationContext",
            system_prompt=(
                "You are an Academic Citation Placer. Given a text draft and a list of references, "
                "you identify exactly where citations should be inserted to support claims. "
                "You format text with proper [Author, Year] brackets. Output JSON."
            )
        )
        self.draft = ""
        self.references = []

    def perceive(self, context: Dict[str, Any]):
        self.draft = context.get("draft_text", "")
        self.references = context.get("references", [])

    def plan(self, goal: str) -> List[str]:
        return ["Analyze text draft for claims", "Match claims with provided references", "Embed citations gracefully"]

    def act(self, instructions: str) -> None:
        prompt = f"""
        Draft Text:
        {self.draft}
        
        Available References:
        {self.references}
        
        Output JSON:
        {{
            "cited_text_md": "The updated draft text with [Author, Year] citations embedded at the correct logic points.",
            "unused_references": ["List of references that didn't fit"]
        }}
        """
        self.current_result = self.llm.chat([{"role": "user", "content": prompt}])

    def verify(self, action_result: Any) -> bool:
        return self.current_result and "cited_text_md" in self.current_result

    def summarize(self) -> Any:
        try:
            import json
            data = json.loads(self.current_result.replace("```json", "").replace("```", "").strip())
        except Exception:
            data = {"raw": self.current_result}
        return self.create_checkpoint("Embed Citations", data, "HIGH")
