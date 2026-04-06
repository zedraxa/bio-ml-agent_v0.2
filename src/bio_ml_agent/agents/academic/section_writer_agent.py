import logging
from typing import Dict, Any, List
from bio_ml_agent.core.agent_base import BaseSubAgent

logger = logging.getLogger("academic.section_writer")

class SectionWriterAgent(BaseSubAgent):
    """
    A2: Section Writer Agent
    Modular writer tailored for specific sections (Introduction, Methods, Results, Discussion, Conclusion).
    """
    def __init__(self, section_name: str):
        super().__init__(
            role_name=f"SectionWriter_{section_name}",
            system_prompt=(
                f"You are an Academic {section_name} Writer for a scientific lab report. "
                f"Your specific task is to write ONLY the '{section_name}' section academically and formally. "
                "If writing Methods, use passive voice. If writing Introduction, state hypothesis clearly. "
                "If Results, narrate the data without interpreting. "
                "Output MUST be in Markdown enclosed within a JSON block under 'content'."
            )
        )
        self.section_name = section_name
        self.input_data: Dict[str, Any] = {}

    def perceive(self, context: Dict[str, Any]):
        self.input_data = context

    def plan(self, goal: str) -> List[str]:
        return [f"Structure the {self.section_name} draft", "Apply scientific tense and vocabulary", "Ensure Markdown formatting in JSON"]

    def act(self, instructions: str) -> None:
        prompt = f"""
        Write the {self.section_name} section using these inputs:
        {self.input_data.get('content_bundle')}
        
        Output a JSON:
        {{
            "section_name": "{self.section_name}",
            "content": "MARKDOWN STRING OF THE WRITTEN SECTION"
        }}
        """
        self.current_result = self.llm.chat([{"role": "user", "content": prompt}])

    def verify(self, action_result: Any) -> bool:
        return self.current_result and self.section_name in self.current_result and "content" in self.current_result

    def summarize(self) -> Any:
        try:
            import json
            data = json.loads(self.current_result.replace("```json", "").replace("```", "").strip())
        except Exception:
            data = {"error": "Failed to parse section output", "raw": self.current_result}

        return self.create_checkpoint(
            action=f"Write {self.section_name} Section",
            data=data,
            confidence="HIGH"
        )
