#!/bin/bash
# Yardımcı betik: Sadece Flask sunucusunu venv ile başlatır.
cd "$(dirname "$0")"

# Port 5000 çakışmasını önle
fuser -k 5000/tcp || true

if [ -f .env ]; then
  # Sadece geçerli satırları export et
  export $(grep -v '^#' .env | xargs)
fi

if [ -d ".venv" ]; then
  source .venv/bin/activate
elif [ -d "venv" ]; then
  source venv/bin/activate
fi

export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"

# Flask ajan uygulamasını başlat
python3 src/bio_ml_agent/whatsapp_connector.py > logs/whatsapp_flask.log 2>&1
