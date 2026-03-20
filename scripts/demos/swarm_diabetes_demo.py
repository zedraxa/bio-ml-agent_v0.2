"""
swarm_diabetes_demo.py
--------------------
Bu betik, Swarm Orchestrator'ı otonom olarak (Pipeline modunda) diyabet verisi üzerinde test eder.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bio_ml_agent.utils.config import load_config
from swarm.orchestrator import SwarmOrchestrator

def main():
    print("=== MULTI-AGENT SWARM DEMO ===")
    
    # Gerçek veri kullanarak pipeline'ı test et
    cfg = load_config()
    cfg.agent.model = "gemini-2.5-flash"  # User configured remote model zzorlaması
    orchestrator = SwarmOrchestrator(cfg)
    
    # Kullanıcıdan gelen "kanser", "analiz et", "uçtan uca" kelimeleri PIPELINE tetikler.
    test_msg = "Buradaki data/raw/diabetes.csv veri setini oku, eksik değerleri temizle, model kur ve analiz et."
    messages = [{"role": "user", "content": test_msg}]
    
    print(f"Kullanıcı İsteği: {test_msg}")
    print("\nOrkestratör Başlatılıyor...\n")
    
    response = orchestrator.process(messages)
    
    print("\n\n=== SWARM TOPLULUĞU FINAL YANITI ===")
    print(response)

if __name__ == "__main__":
    main()
