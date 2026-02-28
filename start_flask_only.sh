#!/bin/bash
# Yardımcı betik: Sadece Flask sunucusunu venv ile başlatır.
cd "$(dirname "$0")"

if [ -f .env ]; then
  export $(cat .env | xargs)
fi

if [ -d "venv" ]; then
  source venv/bin/activate
  
  # Oto-kurulum
  if ! python -c "import twilio" &> /dev/null; then
    pip install twilio flask
  fi
fi

# Flask ajan uygulamasını başlat
python whatsapp_connector.py > logs/whatsapp_flask.log 2>&1
