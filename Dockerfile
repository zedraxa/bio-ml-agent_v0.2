# Bio-ML Agent Dockerfile
# v8: Security-Hardened (non-root user, minimal layers)

FROM python:3.11-slim

# Çalışma dizini
WORKDIR /app

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get install -y --no-install-recommends \
    chromium \
    libatk-bridge2.0-0 \
    libgbm1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root kullanıcı oluştur (güvenlik)
RUN groupadd -r agent && useradd -r -g agent -d /app -s /sbin/nologin agent

# Proje dosyalarını kopyala
COPY pyproject.toml .
COPY . .

# pyproject.toml ile bağımlılıkları kur
RUN pip install --no-cache-dir ".[all,ml_ops,cloud,xai]"

# WhatsApp client bağımlılıklarını kur
RUN cd whatsapp-client && npm install

# Log ve workspace klasörlerini hazırla ve sahipliği ayarla
RUN mkdir -p logs workspace mlflow_logs db && \
    chown -R agent:agent /app

# Ortam değişkenleri
ENV PYTHONUNBUFFERED=1
ENV WORKSPACE_DIR=/app/workspace
ENV LOG_DIR=/app/logs

# Non-root kullanıcıya geç
USER agent

# API Portu (api_server.py ile aynı)
EXPOSE 8001

# Sağlık kontrolü
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Varsayılan: API Server
CMD ["python3", "api_server.py"]
