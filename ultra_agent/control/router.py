import logging
from typing import Dict, Any, Optional
import os

log = logging.getLogger("bio_ml_agent")

class LLMRouter:
    """
    S5-1, S5-2, S5-3, S5-4: 
    LiteLLM bazlı Model Yönlendirme ve Politika Kontrolcüsü.
    """
    def __init__(self):
        # LiteLLM gateway url
        self.gateway_url = os.getenv("LITELLM_GATEWAY_URL", "http://litellm:4000")
        
    def _evaluate_task_complexity(self, system_prompt: str, user_prompt: str) -> str:
        """
        Görevin zorluk seviyesini metin bazlı analiz eder (basit heuristic).
        """
        combined = (system_prompt + " " + user_prompt).lower()
        if any(kw in combined for kw in ["python", "bash", "kodla", "tasarla", "mimari"]):
            return "high"
        elif any(kw in combined for kw in ["analiz", "özet", "çeviri", "açıkla"]):
            return "medium"
        return "low"
        
    def route_request(self, system_prompt: str, user_prompt: str, user_tier: str = "standard") -> Dict[str, Any]:
        """
        S5-2: Routing Politikası
        """
        complexity = self._evaluate_task_complexity(system_prompt, user_prompt)
        
        # Basit routing kararları
        if complexity == "high":
            model = "claude-3-opus-20240229"
            fallback = "gpt-4-turbo"
        elif complexity == "medium":
            model = "claude-3-sonnet-20240229"
            fallback = "gpt-3.5-turbo"
        else:
            # Low complexity: Local model
            model = "ollama/llama3"
            fallback = "gemini-1.5-flash"
            
        routing_decision = {
            "primary_model": model,
            "fallback_model": fallback,
            "complexity_level": complexity,
            "endpoint": self.gateway_url,
            # S5-4: Data Residency test
            "inference_geo": "eu" if os.getenv("GDPR_STRICT") == "true" else "us"
        }
        
        log.info(f"Yönlendirme kararı: {model} (Fallback: {fallback}), Coğrafi: {routing_decision['inference_geo']}")
        return routing_decision
        
    def track_cost(self, agent_id: str, tokens_used: int, estimated_cost: float):
        """
        S5-3: Cost Tracking (Basit Log/Redis stub)
        """
        log.info(f"[Maliyet Faturası] Ajan: {agent_id} | Token: {tokens_used} | Maliyet: ${estimated_cost:.4f}")
        # Gerçek kodda Redis ya da PostgreSQL'e Cost entity'si olarak insert edilir.
        # Örnek limit mekanizması:
        # if total_cost > budget:
        #     raise CostLimitExceededError(...)
