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

    def _evaluate_task_tier(self, system_prompt: str, user_prompt: str) -> int:
        """
        Görevin zorluk seviyesini metin bazlı formata (1, 2, 3 tier) göre analiz eder.
        """
        combined = (system_prompt + " " + user_prompt).lower()
        if any(kw in combined for kw in ["python", "bash", "kodla", "tasarla", "mimari", "sistem", "algoritma"]):
            return 3 # High
        elif any(kw in combined for kw in ["analiz", "özet", "çeviri", "açıkla", "ara", "dosya"]):
            return 2 # Medium
        return 1 # Low

    def _detect_required_caps(self, user_prompt: str) -> list[str]:
        """Kullanıcının isteğine göre gereken minimum özellikleri belirler."""
        caps = []
        combined = user_prompt.lower()
        if any(kw in combined for kw in ["resim", "resm", "görsel", "foto", "image"]):
            caps.append("vision")
        if any(kw in combined for kw in ["ses", "audio", "dinle"]):
            caps.append("audio")
        if any(kw in combined for kw in ["belge yükle"]):
            caps.append("file_upload")
        # Ajanik görevler genellikle tool kullanımına güvenir
        caps.append("tool_use")
        return caps

    def route_request(self, system_prompt: str, user_prompt: str, user_tier: str = "standard") -> Dict[str, Any]:
        """
        S5-2: Dinamik Yönlendirme Politikası
        ModelCapability Registry'i kullanarak eldeki provider'lara göre en iyi 
        ve en mantıklı modeli seçer. Hardcoded model atamaları yerine 
        sistematik uzaklık skoru kullanılır.
        """
        target_tier = self._evaluate_task_tier(system_prompt, user_prompt)
        required_caps = self._detect_required_caps(user_prompt)

        # Mevcut API anahtarlarını kontrol et ve provider listesini oluştur
        available_providers = []
        if os.getenv("OPENAI_API_KEY"): available_providers.append("openai")
        if os.getenv("ANTHROPIC_API_KEY"): available_providers.append("anthropic")
        if os.getenv("GEMINI_API_KEY"): available_providers.append("gemini")

        try:
            from bio_ml_agent.llm_backend import filter_and_sort_models
            models = filter_and_sort_models(required_caps, target_tier, available_providers)

            primary_model = models[0] if models else "qwen2.5"
            fallback_model = models[1] if len(models) > 1 else primary_model

        except ImportError:
            # Fallback fail-safe: Eğer circular import vb yaşanırsa manuel local modele dön
            primary_model = "qwen2.5"
            fallback_model = "llama3"

        complexity_label = {1: "low", 2: "medium", 3: "high"}.get(target_tier, "medium")

        routing_decision = {
            "primary_model": primary_model,
            "fallback_model": fallback_model,
            "complexity_level": complexity_label,
            "target_tier": target_tier,
            "required_caps": required_caps,
            "endpoint": self.gateway_url,
            # S5-4: Data Residency test
            "inference_geo": "eu" if os.getenv("GDPR_STRICT") == "true" else "us"
        }

        log.info(f"Dinamik Yönlendirme: {primary_model} (Fallback: {fallback_model}) | Tier: {target_tier} | Caps: {required_caps}")
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
