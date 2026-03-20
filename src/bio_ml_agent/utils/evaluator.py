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
    
    def __init__(self, registry_file: str = "capability_registry.json"):
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

