#!/bin/bash
# scripts/run_tests.sh
set -e

# Proje kökünü PYTHONPATH'e ekle
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

echo "🚀 Bio-ML Agent Test Suite başlatılıyor..."

# 1. Smoke Tests (Hızlı kontrol)
echo "🔍 Smoke testler çalıştırılıyor..."
./.venv/bin/python3 -m pytest tests/test_smoke.py -v

# 2. Integration Tests
echo "🔗 Entegrasyon testleri çalıştırılıyor..."
./.venv/bin/python3 -m pytest tests/test_agent_integration.py -v

# 3. Diğer testler (Opsiyonel: Hepsini çalıştır)
# pytest tests/ -v

echo "✅ Tüm testler başarıyla tamamlandı!"
