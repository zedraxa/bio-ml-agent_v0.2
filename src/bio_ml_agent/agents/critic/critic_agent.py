import logging
import json
import uuid
from typing import List, Dict, Any, Optional
from bio_ml_agent.core.agent_base import BaseSubAgent
from bio_ml_agent.brain.models import AgentResult, Confidence, Evidence, Comment, TargetType, IntentType, Severity, ReviewerRole, ReviewMode
from bio_ml_agent.llm_backend import auto_create_backend

log = logging.getLogger("critic_agent")

class CriticAgent(BaseSubAgent):
    """
    CriticAgent (Professional):
    E2/R7-D2: Generic Critic Agent.
    Evaluates other agents' outputs and directly injects inline Comments into the artifact.
    """
    
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        # The BaseSubAgent expects only model_name, not role and model_name
        super().__init__(model_name=model_name)
        self.role_name = "CriticAgent"
        self.llm = auto_create_backend(model_name)
        self.last_review: str = ""
        self.critic_result: Dict[str, Any] = {}
        self.generated_comments: List[Comment] = []

    def perceive(self, context: Dict[str, Any]) -> None:
        self.target_result = context.get("target_result") # AgentResult nesnesi
        self.mission_goal = context.get("goal")
        self.artifact_id = context.get("artifact_id", "unknown_artifact")

    def plan(self, goal: str) -> List[str]:
        return ["Check evidence sufficiency", "Identify logical contradictions", "Score confidence level", "Generate inline comments"]

    def act(self, step: str) -> Any:
        if not self.target_result: return "Nothing to criticize"
        
        prompt = f"""Bir Adversarial Critic'sin. Aşağıdaki sonucu denetle.
Hedef: {self.mission_goal}
Sonuç Verisi: {self.target_result.data}

Bu sonuç güvenilir mi? Kanıtlarda boşluk var mı? Yanıtını sert ve analitik bir dille ver.
Return a structured JSON with two keys:
1. "review_summary": A string summarizing the critique.
2. "inline_comments": A list of specific feedback items.

Format for inline_comments:
[
  {{
    "target": "paragraph 2",
    "content": "Bu claim kanıtsız.",
    "severity": "high",
    "target_type": "paragraph"
  }}
]
"""

        try:
            response = self.llm.chat([{"role": "user", "content": prompt}])
            clean_text = response.replace("```json", "").replace("```", "").strip()
            self.critic_result = json.loads(clean_text)
            self.last_review = self.critic_result.get("review_summary", "")
            
            # D2: Extract inline comments
            inline_comments_data = self.critic_result.get("inline_comments", [])
            for c_data in inline_comments_data:
                try:
                    target_type_str = c_data.get("target_type", "general").upper()
                    if target_type_str == "PARAGRAPH": t_type = TargetType.PARAGRAPH
                    elif target_type_str == "CODE_LINE": t_type = TargetType.CODE_LINE
                    elif target_type_str == "IMAGE_REGION": t_type = TargetType.IMAGE_REGION
                    else: t_type = TargetType.ARTIFACT
                    
                    comment = Comment(
                        comment_id=f"critique_{uuid.uuid4().hex[:6]}",
                        target_id=self.artifact_id,
                        target_type=t_type,
                        content=f"[{c_data.get('target', 'General')}] {c_data.get('content', '')}",
                        author="CriticAgent",
                        role=ReviewerRole.CRITIC_AGENT,
                        review_mode=ReviewMode.SCIENTIFIC,
                        intent=IntentType.CRITIQUE,
                        severity=Severity.HIGH if c_data.get("severity", "") == "high" else Severity.MEDIUM,
                        is_actionable=True
                    )
                    self.generated_comments.append(comment)
                except Exception as e:
                    log.error(f"Failed to parse inline comment from critic: {e}")
                    
            log.info(f"CriticAgent generated {len(self.generated_comments)} inline comments (D2).")
            return "Audit completed"

        except Exception as e:
            log.error(f"Critic LLM hatası veya JSON parsing hatası: {e}\nRaw output: {response if 'response' in locals() else 'None'}")
            return "Audit failed"

    def verify(self, action_result: Any) -> bool:
        return len(self.critic_result) > 0

    def summarize(self) -> AgentResult:
        is_valid = "PASS" in self.last_review.upper() or "TRUSTWORTHY" in self.last_review.upper()
        
        # D2: Attach self_comments directly to the result
        return AgentResult(
            agent_id=self.role_name,
            step_id="critic_review",
            status="COMPLETED",
            success=is_valid,
            data={"critic_report": self.critic_result},
            confidence=Confidence(score=0.9 if is_valid else 0.4),
            evidence=[Evidence(source="criticism", content_snippet=self.last_review[:200])],
            message="Critique completed with adversarial focus.",
            self_comments=self.generated_comments
        )
