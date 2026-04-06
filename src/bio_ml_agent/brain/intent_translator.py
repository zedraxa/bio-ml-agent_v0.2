from typing import List, Dict, Any, Optional
import logging
from .models import AgentRole, TaskType

logger = logging.getLogger(__name__)

# ─── Intent Patterns ──────────────────────────────────────────────────────────

INTENT_PATTERNS = [
    {
        "intent": "mitosis_phase",
        "keywords": ["mitosis", "cell cycle", "phase identification", "microscopy"],
        "primary_agent": AgentRole.MICROSCOPY_AGENT,
        "supporting": [AgentRole.RESEARCHER, AgentRole.CRITIC],
        "description": "Identify mitotic phases from microscopy images."
    },
    {
        "intent": "sequence_analysis",
        "keywords": ["sequence", "fasta", "dna", "rna", "alignment", "blast"],
        "primary_agent": AgentRole.IN_SILICO_EXPERT,
        "supporting": [AgentRole.DATA_ENGINEER, AgentRole.CRITIC],
        "description": "Perform sequence alignment and bio-computational analysis."
    },
    {
        "intent": "full_pipeline",
        "keywords": ["pipeline", "end-to-end", "workflow", "complete system"],
        "primary_agent": AgentRole.ML_EXPERT,
        "supporting": [AgentRole.RESEARCHER, AgentRole.DATA_ENGINEER, AgentRole.CRITIC],
        "description": "Run a comprehensive multi-agent research pipeline."
    },
    {
        "intent": "ml_pipeline",
        "keywords": ["train", "model", "prediction", "neural", "classification"],
        "primary_agent": AgentRole.ML_EXPERT,
        "supporting": [AgentRole.DATA_ENGINEER, AgentRole.CRITIC],
        "description": "Build and train a machine learning model for bio-tasks."
    },
    {
        "intent": "lab_report",
        "keywords": ["report", "write", "summary", "findings", "documentation"],
        "primary_agent": AgentRole.ACADEMIC_EXPERT,
        "supporting": [AgentRole.RESEARCHER, AgentRole.CRITIC],
        "description": "Synthesize results into a professional research report."
    }
]

class IntentTranslator:
    """
    Part VI: Intent Translator Module.
    
    Converts natural language prompts into structured scientific intents
    using keyword mapping and feature gating.
    """

    def __init__(self, patterns: List[Dict[str, Any]] = INTENT_PATTERNS):
        self.patterns = patterns

    def translate(self, prompt: str) -> List[Dict[str, Any]]:
        """Detect one or more intents from the user prompt."""
        from .feature_flags import FEATURE_CONTROLLER # Local import to avoid circularity

        prompt_lower = prompt.lower()
        matched = []

        for p in self.patterns:
            score = sum(1 for kw in p["keywords"] if kw in prompt_lower)
            if score > 0:
                matched.append({**p, "score": score})

        # Sort by match score descending
        matched.sort(key=lambda x: x["score"], reverse=True)

        # If nothing matched, default to full_pipeline
        if not matched:
            matched = [next(p for p in self.patterns if p["intent"] == "full_pipeline")]
            matched[0]["score"] = 1

        # Axis H4: Filter by Feature Flags
        filtered = []
        for it in matched:
            feature_name = f"scenario.{it['intent']}"
            if FEATURE_CONTROLLER.is_enabled(feature_name):
                filtered.append(it)
            else:
                logger.warning(f"[IntentTranslator:H4] Scenario '{it['intent']}' is DISABLED.")

        return filtered if filtered else matched[:1]
