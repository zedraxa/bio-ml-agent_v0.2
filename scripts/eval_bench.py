#!/usr/bin/env python3
# scripts/eval_bench.py
# ═══════════════════════════════════════════════════════════
#  Bio-ML Agent — Model Değerlendirme & Benchmark Aracı
#  Farklı modellerin yanıt kalitesini ve hızını ölçer.
# ═══════════════════════════════════════════════════════════

import time
import argparse
import logging
import json
from pathlib import Path
from typing import List, Dict, Any

# Proje kök dizinini ekle
import sys
sys.path.append(str(Path(__file__).parent.parent))

from llm_backend import auto_create_backend, list_backends, get_model_capabilities
from utils.config import get_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger("eval_bench")

# Test Seti
DEFAULT_TEST_SET = [
    {
        "id": "bio_summarize",
        "category": "Biology",
        "prompt": "DNA replikasyon sürecini 3 adımda teknik terimlerle özetle.",
        "expected_keywords": ["polimeraz", "helikaz", "replikasyon çatalı"]
    },
    {
        "id": "coding_python",
        "category": "Coding",
        "prompt": "Python'da bir listenin medyanını bulan fonksiyonu yaz.",
        "expected_keywords": ["def", "sort", "len"]
    },
    {
        "id": "rag_complex",
        "category": "Reasoning",
        "prompt": "Eğer bir hastanın kan şekeri yüksekse ve insülin direnci varsa, hangi biyobelirteçler takip edilmelidir?",
        "expected_keywords": ["hba1c", "glukoz"]
    }
]

def run_benchmark(models: List[str], test_set: List[Dict[str, Any]], mode: str = "auto"):
    results = []
    
    for model_name in models:
        log.info(f"🚀 Model Test Ediliyor: {model_name}")
        tests_list: List[Dict[str, Any]] = []
        model_results: Dict[str, Any] = {
            "model": model_name,
            "backend": "unknown",
            "tests": tests_list,
            "avg_latency": 0.0,
            "success_rate": 0.0
        }
        
        try:
            backend = auto_create_backend(model_name, mode=mode)
            model_results["backend"] = backend.name
            
            total_latency: float = 0.0
            successful_tests: int = 0
            
            for test in test_set:
                start_time = time.time()
                try:
                    response = backend.chat([{"role": "user", "content": test["prompt"]}])
                    latency = float(time.time() - start_time)
                    total_latency += latency
                    
                    # Basit anahtar kelime kontrolü
                    score_count = 0
                    for kw in test["expected_keywords"]:
                        if kw.lower() in response.lower():
                            score_count += 1
                    
                    tests_list.append({
                        "test_id": test["id"],
                        "latency": latency,
                        "score": score_count / len(test["expected_keywords"]),
                        "success": True
                    })
                    successful_tests += 1
                    log.info(f"  ✅ {test['id']} | Time: {latency:.2f}s | Score: {score_count}/{len(test['expected_keywords'])}")
                    
                except Exception as e:
                    log.error(f"  ❌ {test['id']} Hatası: {e}")
                    tests_list.append({
                        "test_id": test["id"],
                        "error": str(e),
                        "success": False
                    })
            
            if successful_tests > 0:
                model_results["avg_latency"] = float(total_latency / successful_tests)
                model_results["success_rate"] = float(successful_tests / len(test_set))
                
        except Exception as e:
            log.error(f"⚠️ Model {model_name} başlatılamadı: {e}")
            continue
            
        results.append(model_results)
        
    return results

def print_table(results):
    print("\n" + "="*80)
    print(f"{'MODEL':<30} | {'BACKEND':<10} | {'AVG TIME':<10} | {'SUCCESS':<10}")
    print("-" * 80)
    for res in results:
        print(f"{res['model']:<30} | {res['backend']:<10} | {res['avg_latency']:<10.2f}s | {res['success_rate']*100:>8.1f}%")
    print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Bio-ML Agent Model Benchmark")
    parser.add_argument("--models", nargs="+", help="Test edilecek modeller (boşlukla ayırın)")
    parser.add_argument("--mode", default="auto", choices=["auto", "local", "remote"], help="Backend modu")
    parser.add_argument("--output", help="Sonuçları JSON olarak kaydet")
    
    args = parser.parse_args()
    
    config = get_config()
    models = args.models or [config.agent.model]
    
    log.info("📊 Benchmark Başlatılıyor...")
    results = run_benchmark(models, DEFAULT_TEST_SET, mode=args.mode)
    
    print_table(results)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"💾 Sonuçlar kaydedildi: {args.output}")

if __name__ == "__main__":
    main()
