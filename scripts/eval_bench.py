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

from bio_ml_agent.llm_backend import auto_create_backend
from bio_ml_agent.ml.evaluator import AgentEvaluator, CapabilityRegistry
from bio_ml_agent.utils.config import get_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger("eval_bench")

def load_scenarios(category: str = None) -> List[Dict[str, Any]]:
    scenarios_path = Path(__file__).parent.parent / "tests" / "benchmarks" / "scenarios.json"
    if not scenarios_path.exists():
        log.error(f"Senaryo dosyası bulunamadı: {scenarios_path}")
        return []

    try:
        with open(scenarios_path, "r", encoding="utf-8") as f:
            scenarios = json.load(f)
            if category:
                scenarios = [s for s in scenarios if s.get("domain") == category]
            return scenarios
    except Exception as e:
        log.error(f"Senaryolar yüklenirken hata oluştu: {e}")
        return []

def run_benchmark(models: List[str], test_set: List[Dict[str, Any]]):
    results = []
    registry = CapabilityRegistry()
    evaluator = AgentEvaluator(registry=registry)

    workspace_dir = Path(__file__).parent.parent / "workspace" / "benchmark_workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    for model_name in models:
        log.info(f"🚀 Model Test Ediliyor: {model_name}")
        tests_list: List[Dict[str, Any]] = []
        model_results: Dict[str, Any] = {
            "model": model_name,
            "tests": tests_list,
            "avg_latency": 0.0,
            "success_rate": 0.0
        }
        
        try:
            total_latency: float = 0.0
            successful_tests: int = 0
            
            for test in test_set:
                log.info(f"  ▶️ Senaryo: {test['scenario_id']} ({test['name']})")
                try:
                    result = evaluator.evaluate_scenario(model_name, test, workspace=str(workspace_dir))
                    
                    tests_list.append(result)
                    
                    latency = result.get("latency", 0)
                    total_latency += latency
                    
                    if result.get("success"):
                        successful_tests += 1
                        log.info(f"  ✅ {test['scenario_id']} Başarılı | Time: {latency:.2f}s")
                    else:
                        log.warning(f"  ❌ {test['scenario_id']} Başarısız | Time: {latency:.2f}s")
                        
                except Exception as e:
                    log.error(f"  ❌ {test['scenario_id']} Hatası: {e}")
                    tests_list.append({
                        "scenario_id": test["scenario_id"],
                        "error": str(e),
                        "success": False
                    })
            
            if len(test_set) > 0:
                model_results["avg_latency"] = float(total_latency / len(test_set))
                model_results["success_rate"] = float(successful_tests / len(test_set))
                
            registry.update_benchmark_score(model_name, model_results["success_rate"])
                
        except Exception as e:
            log.error(f"⚠️ Model {model_name} değerlendirilemedi: {e}")
            continue
            
        results.append(model_results)
        
    return results

def print_table(results):
    print("\n" + "="*80)
    print(f"{'MODEL':<30} | {'AVG TIME':<10} | {'SUCCESS':<10}")
    print("-" * 80)
    for res in results:
        print(f"{res['model']:<30} | {res['avg_latency']:<10.2f}s | {res['success_rate']*100:>8.1f}%")
    print("="*80 + "\n")

def generate_markdown_report(results, output_path: str):
    lines = [
        "# Bio-ML Agent Benchmark Report",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "| Model | Avg Latency | Success Rate |",
        "|-------|-------------|--------------|"
    ]
    for res in results:
        lines.append(f"| {res['model']} | {res['avg_latency']:.2f}s | {res['success_rate']*100:.1f}% |")
    
    lines.extend([
        "",
        "## Detailed Results"
    ])
    
    for res in results:
        lines.extend([
            f"### Model: {res['model']}",
            "| Scenario | Success | Latency | Expected Tools | Matched Tools | Content Matched | No Forbidden |",
            "|----------|---------|---------|----------------|---------------|-----------------|--------------|"
        ])
        for t in res['tests']:
            s_name = t.get("scenario", "unknown")
            s_ok = "✅" if t.get("success") else "❌"
            s_lat = f"{t.get('latency', 0):.2f}s"
            t_exp = ", ".join(t.get("tools_expected", [])) or "None"
            t_match = "✅" if t.get("tools_matched") else "❌"
            c_match = "✅" if t.get("content_matched") else "❌"
            nf_match = "✅" if t.get("content_no_forbidden") else "❌"
            lines.append(f"| {s_name} | {s_ok} | {s_lat} | {t_exp} | {t_match} | {c_match} | {nf_match} |")
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info(f"📄 Markdown raporu oluşturuldu: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Bio-ML Agent Model Benchmark")
    parser.add_argument("--models", nargs="+", help="Test edilecek modeller (boşlukla ayırın)")
    parser.add_argument("--category", help="Sadece belirli domain/kategorideki senaryoları çalıştır", default=None)
    parser.add_argument("--output", help="Sonuçları JSON olarak kaydet", default=None)
    parser.add_argument("--report", help="Sonuçları Markdown olarak kaydet", default="benchmark_report.md")
    
    args = parser.parse_args()
    
    config = get_config()
    models = args.models or [config.agent.model]
    
    scenarios = load_scenarios(category=args.category)
    if not scenarios:
        log.error("Çalıştırılacak senaryo bulunamadı.")
        sys.exit(1)
        
    log.info(f"📊 Benchmark Başlatılıyor... ({len(scenarios)} senaryo, {len(models)} model)")
    results = run_benchmark(models, scenarios)
    
    print_table(results)
    
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        log.info(f"💾 JSON kayıt edildi: {args.output}")
        
    if args.report:
        generate_markdown_report(results, args.report)

if __name__ == "__main__":
    main()
