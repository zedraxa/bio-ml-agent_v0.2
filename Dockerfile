# ═══════════════════════════════════════════════════════════
#  Bio-ML Agent — Dockerfile
# ═══════════════════════════════════════════════════════════

FROM python:3.11-slim

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizini
WORKDIR /app

# Bağımlılıkları önce kopyala (cache optimizasyonu)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Proje dosyalarını kopyala
COPY . .

# Workspace klasörü oluştur
RUN mkdir -p workspace logs

# Ortam değişkenleri
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Varsayılan komut
CMD ["python3", "agent.py", "--model", "gemini-2.5-flash"]
