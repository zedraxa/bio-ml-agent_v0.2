# Bio-ML Agent Dockerfile
# Sprint 6: Deployment & Portability

FROM python:3.11-slim

# Çalışma dizini
WORKDIR /app

# Sistem bağımlılıkları (psutil için gcc vb.)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Proje dosyalarını kopyala
COPY . .

# Bağımlılıkları kopyala ve kur
RUN pip install --no-cache-dir ".[all,ml_ops,cloud]"

# Log ve workspace klasörlerini hazırla
RUN mkdir -p logs workspace mlflow_logs db

# Ortam değişkenleri
ENV PYTHONUNBUFFERED=1
ENV WORKSPACE_DIR=/app/workspace
ENV LOG_DIR=/app/logs

# API Portu
EXPOSE 8000

# Varsayılan olarak API Server'ı başlat
# (Ajan CLI modunda kullanılmak istenirse docker run ile ezilebilir)
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
