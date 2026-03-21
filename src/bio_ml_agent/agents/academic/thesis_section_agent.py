import logging
from typing import Dict, Any
from bio_ml_agent.core.agent_base import BaseSubAgent

logger = logging.getLogger("academic.thesis_section_agent")

class ThesisSectionAgent(BaseSubAgent):
    """
    F3: Thesis Section Agent
    Expands a succinct paper into an elaborate, explanatory thesis chapter.
    """
    def __init__(self):
        super().__init__(
            role_name="ThesisSectionWriter",
            system_prompt=(
                "You are a Thesis Supervisor AI. You take a highly compressed journal article "
                "draft and expand it into a full, pedantic, highly explanatory Thesis Chapter. "
                "You add extensive background context and detailed methodological reasoning. Output JSON."
            )
        )
        self.paper_draft = ""

    def perceive(self, context: Dict[str, Any]):
        self.paper_draft = context.get("paper_draft", "")
        self.chapter_type = context.get("chapter_type", "General")

    def plan(self) -> str:
        return f"Expand paper into a comprehensive Thesis {self.chapter_type} chapter."

    def act(self, instructions: str) -> None:
        prompt = f"""
        Expand this concise article text into a broad Thesis Chapter ({self.chapter_type}):
        {self.paper_draft}
        
        Generate JSON:
        {{
            "thesis_chapter_md": "The heavily expanded, informative markdown section",
            "added_context_flags": ["What background or detail was added"]
        }}
        """
        self.current_result = self.llm.chat([{"role": "user", "content": prompt}])

    def verify(self) -> bool:
        return "thesis_chapter_md" in self.current_result

    def summarize(self) -> Any:
        try:
            import json
            data = json.loads(self.current_result.replace("```json", "").replace("```", "").strip())
        except Exception:
            data = {"raw": self.current_result}
        return self.create_checkpoint("Draft Thesis Chapter", data, "HIGH")
