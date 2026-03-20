import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class CapabilityRegistry:
    """
    LLM modellerinin yeteneklerini (metin, görüntü okuma, uzun bağlam vs.) ve
    geçmiş testlerdeki (benchmark) başarı metriklerini takip eder.
    Ajan, belirli bir görev için en uygun modeli Capability Registry'ye sorarak seçebilir.
    """
    
    def __init__(self, registry_file: str = None):
        if registry_file is None:
            # src/bio_ml_agent/ml/evaluator.py -> src/bio_ml_agent/resources/
            registry_file = Path(__file__).parent.parent / "resources" / "capability_registry.json"
        self.registry_file = Path(registry_file)
        # Varsayılan yetenek matrisi
        self.registry: Dict[str, Dict[str, Any]] = {
            "gemini-2.5-flash": {
                "multimodal": True,
                "context_window": 1000000,
                "speed": "fast",
                "cost": "low",
                "tool_use_accuracy": 0.95,
                "benchmark_score": 0.0
            },
            "gemini-2.0-pro-exp-02-05": {
                "multimodal": True,
                "context_window": 2000000,
                "speed": "moderate",
                "cost": "high",
                "tool_use_accuracy": 0.98,
                "benchmark_score": 0.0
            },
            "gpt-4o": {
                "multimodal": True,
                "context_window": 128000,
                "speed": "fast",
                "cost": "high",
                "tool_use_accuracy": 0.97,
                "benchmark_score": 0.0
            },
            "ollama-llama-3": {
                "multimodal": False,
                "context_window": 8000,
                "speed": "varies",
                "cost": "free",
                "tool_use_accuracy": 0.75,
                "benchmark_score": 0.0
            }
        }
        self.load()

    def load(self):
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Mevcut configi ezmeyip güncelleyelim
                    for k, v in data.items():
                        if k in self.registry:
                            self.registry[k].update(v)
                        else:
                            self.registry[k] = v
            except Exception as e:
                logger.warning(f"Kayıt yüklenemedi: {e}")

    def save(self):
        try:
            with open(self.registry_file, 'w', encoding='utf-8') as f:
                json.dump(self.registry, f, indent=4)
        except Exception as e:
            logger.error(f"Kayıt dosyası yazılamadı: {e}")

    def update_benchmark_score(self, model: str, score: float):
        """Çalıştırılan sentetik testler sonrası kalite metriklerini günceller."""
        if model not in self.registry:
            self.registry[model] = {"multimodal": False, "benchmark_score": 0.0}
        self.registry[model]["benchmark_score"] = score
        self.save()
        logger.info(f"[Capability Registry] {model} skoru güncellendi: {score}")

    def suggest_best_model(self, requirements: List[str]) -> str:
        """
        Gelen ihtiyaca ('multimodal', 'high_accuracy') göre en uygun modeli önerir.
        Basit skorlama tabanlıdır.
        """
        best_model = None
        best_score = -1.0
        
        for model, caps in self.registry.items():
            score = 0
            if "multimodal" in requirements and not caps.get("multimodal", False):
                continue
            
            # Tool use ve genel benchmark ortalaması
            acc = caps.get("tool_use_accuracy", 0.0)
            bench = caps.get("benchmark_score", 0.0)
            score = acc + bench
            
            if score > best_score:
                best_score = score
                best_model = model
                
        return best_model or "gemini-2.5-flash" # fallback

class BenchmarkHarness:
    """
    LLM'lere sentetik görevler gönderip (ör: 'Python ile 2+2 hesapla')
    beklenen aracı (<PYTHON>) seçip seçmediğini ve doğru formatta
    (JSON, tag vs.) dönüp dönmediğini test eder.
    """
    def __init__(self, registry: CapabilityRegistry):
        self.registry = registry
        self.synthetic_tasks = [
            {
                "prompt": "Sadece Python aracı kullanarak 2+2 işleminin sonucunu hesapla. Cümle kurma.",
                "expected_tool": "<PYTHON>",
                "must_contain": ["2", "+", "2"]
            },
            {
                "prompt": "Bana Bash aracı ile 'pwd' komutunu çalıştıracak bir çıktı üret.",
                "expected_tool": "<BASH>",
                "must_contain": ["pwd"]
            }
        ]

    def run_eval(self, model: str) -> float:
        from bio_ml_agent.llm_backend import auto_create_backend
        
        logger.info(f"[Benchmark Harness] {model} için test başlatılıyor...")
        try:
            backend = auto_create_backend(model)
        except Exception as e:
            logger.error(f"Backend oluşturulamadı: {e}")
            return 0.0

        successes = 0
        total = len(self.synthetic_tasks)

        for task in self.synthetic_tasks:
            try:
                response = backend.chat([{"role": "user", "content": task["prompt"]}])
                
                # Basit doğrulama (Eval logic)
                tool_passed = task["expected_tool"] in response
                content_passed = all(k in response for k in task["must_contain"])
                
                if tool_passed and content_passed:
                    successes += 1
            except Exception as e:
                logger.warning(f"Task '{task['prompt']}' failed for {model}: {e}")

        final_score = successes / total if total > 0 else 0.0
        self.registry.update_benchmark_score(model, final_score)
        return final_score


class AgentEvaluator:
    """
    Bio-ML Agent'ı tam teşekküllü AgentService üzerinden çalıştırıp
    araç kullanımı, mantık ve nihai çıktısını değerlendirir.
    """
    def __init__(self, registry: CapabilityRegistry):
        self.registry = registry

    def evaluate_scenario(self, model: str, scenario: dict, workspace: str) -> dict:
        import time
        from bio_ml_agent.services.agent_service import AgentService

        service = AgentService(model=model, workspace=workspace, timeout=30, max_steps=10)
        # Otomatik mode (kullanıcı onayı bekleme)
        service.approval_mode = 1

        prompt = scenario.get("input_data", {}).get("prompt", "")
        if not prompt:
            return {"success": False, "error": "Prompt bulunamadı"}

        start_time = time.time()
        events = []
        try:
            for event in service.process_message(prompt):
                if event.get("type") in ["tool_output", "tool_start", "assistant", "error"]:
                    events.append(event)
        except Exception as e:
            logger.error(f"Scenario error: {e}")
            return {"success": False, "error": str(e)}

        latency = time.time() - start_time

        # Gerekli araçlar kullanılmış mı kontrol et
        used_tools = [e.get("tool") for e in events if e.get("type") == "tool_start" and e.get("tool")]
        expected_tools = scenario.get("expected_tools", [])
        tools_matched = all(et in used_tools for et in expected_tools)

        # Sonuç metninde arama
        assistant_texts = [e.get("content", "") for e in events if e.get("type") == "assistant" and e.get("content")]
        full_assistant_text = " ".join(assistant_texts)

        must_contain = scenario.get("must_contain", [])
        must_not_contain = scenario.get("must_not_contain", [])

        content_matched_all = all(mc.lower() in full_assistant_text.lower() for mc in must_contain)
        content_no_forbidden = all(mnc.lower() not in full_assistant_text.lower() for mnc in must_not_contain)

        success = tools_matched and content_matched_all and content_no_forbidden

        return {
            "scenario": scenario.get("scenario_id", "unknown"),
            "model": model,
            "success": success,
            "latency": latency,
            "tools_used": used_tools,
            "tools_expected": expected_tools,
            "tools_matched": tools_matched,
            "content_matched": content_matched_all,
            "content_no_forbidden": content_no_forbidden
        }
