import logging
from typing import List, Dict, Any

log = logging.getLogger("capability_registry")

class CapabilityRegistry:
    """
    Capability Registry: LLM modellerinin yeteneklerini (Reasoning, Vision, Tools) 
    ve maliyet/hız profillerini tutar.
    """
    
    def __init__(self):
        self.models = {
            "gemini-2.0-flash": {
                "capabilities": ["vision", "json", "fast", "scout"],
                "reasoning_score": 8,
                "token_limit": 1000000
            },
            "gemini-2.0-pro": {
                "capabilities": ["complex_reasoning", "long_context", "research"],
                "reasoning_score": 10,
                "token_limit": 2000000
            }
        }

    def select_model(self, requirements: List[str]) -> str:
        """İhtiyaçlara göre en uygun modeli seçer."""
        for model_id, profile in self.models.items():
            if all(req in profile["capabilities"] for req in requirements):
                return model_id
        return "gemini-2.0-flash" # Default
