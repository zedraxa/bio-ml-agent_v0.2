#!/bin/bash
# Bio-ML Agent — WhatsApp Gateway Unified Launcher
# Phase 5, Step 6 Integration

set -e

# Renkler
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}===== Bio-ML Agent WhatsApp Gateway =====${NC}"

# 1. Bağımlılık Kontrolleri
echo -e "${YELLOW}[1/3] Sistem Kontrolleri...${NC}"

if ! command -v node &> /dev/null; then
    echo -e "${RED}Hata: Node.js yüklü değil!${NC}"
    exit 1
fi

if ! command -v ./.venv/bin/python &> /dev/null; then
    echo -e "${RED}Hata: Python sanal ortamı (.venv) bulunamadı!${NC}"
    exit 1
fi

# 2. Node.js Bağımlılıklarını Kontrol Et
if [ ! -d "whatsapp-client/node_modules" ]; then
    echo -e "${YELLOW}Node.js paketleri eksik, yükleniyor...${NC}"
    cd whatsapp-client && npm install && cd ..
fi

# 3. Servisleri Başlat
echo -e "${YELLOW}[2/3] Servisler Başlatılıyor...${NC}"

# Python Connector'ı arka planda başlat
echo -e "${GREEN}-> Python WhatsApp Connector başlatılıyor (Port 5000)...${NC}"
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
./.venv/bin/python src/bio_ml_agent/whatsapp_connector.py > logs/whatsapp_py.log 2>&1 &
PY_PID=$!

# Node.js Client'ı ön planda başlat (QR kodunu görmek için)
echo -e "${GREEN}-> Node.js WhatsApp Client başlatılıyor (Port 3001)...${NC}"
echo -e "${YELLOW}DİKKAT: ToS onaylanarak başlatılıyor. QR kodunu taratmanız gerekebilir.${NC}"

# Kapanışta Python sürecini de öldür
trap "kill $PY_PID; echo -e '\n${RED}İptal ediliyor...${NC}'; exit" SIGINT SIGTERM

cd whatsapp-client
node index.js --accept-tos

echo -e "${BLUE}=========================================${NC}"
