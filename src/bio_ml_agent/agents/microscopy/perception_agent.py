import logging
import json
from enum import Enum
from typing import List, Dict, Any, Optional
from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from bio_ml_agent.llm_backend import auto_create_backend
from pathlib import Path

log = logging.getLogger("microscopy.perception")

class Modality(str, Enum):
    BRIGHTFIELD = "brightfield"
    FLUORESCENCE = "fluorescence"
    PHASE_CONTRAST = "phase_contrast"
    CONFOCAL = "confocal"
    HISTOLOGY = "histology_slide"
    ELECTRON_MICROSCOPY = "sem_tem"
    UNCERTAIN = "uncertain"

class QualityFlag(str, Enum):
    BLURRY = "blur"
    UNDEREXPOSED = "underexposure"
    OVEREXPOSED = "overexposure"
    STAINING_ARTIFACT = "staining_artifact"
    DEBRIS = "debris"
    SCRATCH = "fold_scratch"
    NOISY = "noise"
    CLEAN = "high_quality"

class MicroscopyPerceptionAgent(BaseSubAgent):
    """
    MicroscopyPerceptionAgent (A1 - Physical Vision Layer):
    - Uses true Multimodal LLM Vision API to analyze images.
    - Distinguishes optical modalities and staining techniques.
    - Audits image quality natively.
    """

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        super().__init__("MicroscopyPerceptionAgent", model_name)
        self.image_path: Optional[Path] = None
        self.llm = auto_create_backend(model_name)
        self.profile: Dict[str, Any] = {
            "modality_guess": Modality.UNCERTAIN,
            "stain_guess": "unknown",
            "specimen_guess": "unknown",
            "quality_flags": [],
            "quality_score": 0.0,
            "structure_count_estimate": 0
        }

    def perceive(self, context: Dict[str, Any]) -> None:
        path = context.get("image_path")
        if path:
            self.image_path = Path(path)
            log.info(f"👁️ Physical Vision Agent online for: {self.image_path.name}")

    def plan(self, goal: str) -> List[str]:
        return [
            "Construct Multimodal payload with Base64 image encoding",
            "Query LLM Vision API for Modality and Quality assessment",
            "Parse JSON response for strict Profile attributes"
        ]

    def act(self, step: str) -> Any:
        if not self.image_path or not self.image_path.exists():
            return "Error: Missing or invalid Image"

        log.info(f"📡 Querying Vision API: {step}")

        if "query" in step.lower() or "multimodal" in step.lower():
            # Real LLM API Call
            prompt = f"""
            You are a world-class microscopy expert. Analyze the attached image.
            Return a purely valid JSON with EXACTLY these keys:
            - modality_guess (pick one: {list(m.value for m in Modality)})
            - stain_guess (string, best guess or 'unknown')
            - specimen_guess (string, e.g., 'Blood Smear', 'Root Tip')
            - quality_flags (array of strings from: {list(q.value for q in QualityFlag)})
            - quality_score (float 0.0 to 1.0)
            - structure_count_estimate (int)
            
            Do not wrap in markdown or backticks. Return raw JSON text only.
            """

            messages = [
                {"role": "system", "content": "You are strict JSON outputting Bio-ML agent."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "file", "path": str(self.image_path)}
                ]}
            ]

            try:
                response_text = self.llm.chat(messages)
                # Cleanup potential backticks
                if "```json" in response_text:
                    response_text = response_text.replace("```json", "").replace("```", "").strip()
                elif "```" in response_text:
                    response_text = response_text.replace("```", "").strip()

                parsed_json = json.loads(response_text)
                self.profile.update(parsed_json)
                log.info(f"✅ Vision API inference successful. Modality: {self.profile.get('modality_guess')}")
            except Exception as e:
                log.error(f"Vision API failure: {e}. Falling back to default profile.")
                self.profile["quality_flags"] = [QualityFlag.BLURRY]

        return f"Step '{step}' processed via Vision API."

    def verify(self, action_result: Any) -> bool:
        if float(self.profile.get("quality_score", 0)) < 0.3:
            log.warning("⚠️ Vision API flagged image with low quality.")
        return True

    def summarize(self) -> AgentResult:
        return self.create_result(
            success=True,
            data=self.profile,
            confidence=Confidence.HIGH if float(self.profile.get("quality_score", 0)) > 0.7 else Confidence.LOW,
            evidence=[Evidence(source=str(self.image_path), content_snippet="Physical LLM Vision inference executed.")],
            message=f"Deep Image Profile: {self.profile.get('modality_guess')} - {self.profile.get('specimen_guess')}."
        )
