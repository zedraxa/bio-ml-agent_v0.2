# Bio-ML Agent Dockerfile
# v7: Swarm + XAI + Active Learning

FROM python:3.11-slim

# Çalışma dizini
WORKDIR /app

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Proje dosyalarını kopyala
COPY pyproject.toml .
COPY . .

# pyproject.toml ile bağımlılıkları kur (requirements.txt yok)
RUN pip install --no-cache-dir ".[all,ml_ops,cloud,xai]"

# Log ve workspace klasörlerini hazırla
RUN mkdir -p logs workspace mlflow_logs db

# Ortam değişkenleri
ENV PYTHONUNBUFFERED=1
ENV WORKSPACE_DIR=/app/workspace
ENV LOG_DIR=/app/logs

# API Portu (api_server.py ile aynı)
EXPOSE 8001

# Varsayılan: API Server (docker-compose'da override edilir)
CMD ["python3", "api_server.py"]
