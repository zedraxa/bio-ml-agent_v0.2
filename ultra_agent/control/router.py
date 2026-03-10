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
        
        # Mevcut API anahtarlarını kontrol et
        has_openai = bool(os.getenv("OPENAI_API_KEY"))
        has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
        has_gemini = bool(os.getenv("GEMINI_API_KEY"))

        # Zorluk derecesine göre ideal modelleri belirle
        if complexity == "high":
            if has_anthropic:
                model = "claude-3-5-sonnet-20241022"
                fallback = "gpt-4o" if has_openai else "gemini-2.5-pro"
            elif has_openai:
                model = "gpt-4o"
                fallback = "gemini-2.5-pro" if has_gemini else "ollama/qwen2.5:7b-instruct"
            else:
                model = "gemini-2.5-flash" if has_gemini else "ollama/qwen2.5:7b-instruct"
                fallback = "gemini-2.5-flash"
        elif complexity == "medium":
            if has_openai:
                model = "gpt-4o-mini"
                fallback = "gemini-2.5-flash" if has_gemini else "ollama/qwen2.5:7b-instruct"
            elif has_anthropic:
                model = "claude-3-haiku-20240307"
                fallback = "gemini-2.5-flash" if has_gemini else "ollama/qwen2.5:7b-instruct"
            else:
                model = "gemini-2.5-flash" if has_gemini else "ollama/qwen2.5:7b-instruct"
                fallback = "gemini-1.5-flash"
        else:
            # Low complexity: Local model veya çok hızlı modeller
            if has_gemini:
                model = "gemini-2.5-flash"
                fallback = "ollama/llama3"
            else:
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
