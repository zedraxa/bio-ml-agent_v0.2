#!/bin/bash
# Bio-ML Agent — Temporal Worker Launcher
# Phase 5, Step 7 Integration

set -e

# Renkler
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}===== Bio-ML Agent Temporal Worker =====${NC}"

# 1. Kontroller
if ! command -v ./.venv/bin/python &> /dev/null; then
    echo -e "${RED}Hata: Python sanal ortamı (.venv) bulunamadı!${NC}"
    exit 1
fi

# 2. Çevresel Değişkenler
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# 3. Başlat
echo -e "${YELLOW}Temporal Worker başlatılıyor...${NC}"
echo -e "${GREEN}Queue: bio-ml-queue${NC}"
echo -e "${GREEN}Workflows: WorkspaceIndexing, VirtualScreening${NC}"

# Temporal server kontrolü (opsiyonel ama bilgilendirici)
if ! nc -z localhost 7233 2>/dev/null; then
    echo -e "${RED}UYARI: localhost:7233 portuna erişilemiyor.${NC}"
    echo -e "${YELLOW}Lütfen Temporal server'ın (Docker veya dev-server) çalıştığından emin olun.${NC}"
fi

./.venv/bin/python src/bio_ml_agent/ultra_agent/orchestration/temporal_workflows/worker.py
