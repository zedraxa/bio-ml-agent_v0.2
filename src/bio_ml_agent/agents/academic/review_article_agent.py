import logging
from typing import Dict, Any
from bio_ml_agent.core.agent_base import BaseSubAgent

logger = logging.getLogger("academic.review_article_agent")

class ReviewArticleAgent(BaseSubAgent):
    """
    B5: Review Article Agent
    Generates thematic subsections from a bundle of literature sources.
    """
    def __init__(self):
        super().__init__(
            role_name="ReviewArticleGenerator",
            system_prompt=(
                "You are an expert Review Article Writer. "
                "Given a cluster of related papers, you deduce core themes, contrast findings, "
                "and generate a cohesive review outline featuring synthetic paragraphs."
                "Output JSON."
            )
        )
        self.papers = []

    def perceive(self, context: Dict[str, Any]):
        self.papers = context.get("literature_bundle", [])

    def plan(self) -> str:
        return "Find common themes in literature and draft review article."

    def act(self, instructions: str) -> None:
        prompt = f"""
        Synthesize the following literature bundle into a Review Article text:
        {self.papers}
        
        Produce JSON:
        {{
            "review_text_md": "Markdown text with headers and citations",
            "thematic_clusters": ["Theme 1", "Theme 2"],
            "conflicting_evidence": "Notes on where papers disagree"
        }}
        """
        self.current_result = self.llm.chat([{"role": "user", "content": prompt}])

    def verify(self) -> bool:
        return "review_text_md" in self.current_result

    def summarize(self) -> Any:
        try:
            import json
            data = json.loads(self.current_result.replace("```json", "").replace("```", "").strip())
        except Exception:
            data = {"raw": self.current_result}
        return self.create_checkpoint("Write Review Article", data, "HIGH")
