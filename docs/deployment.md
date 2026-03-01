# Bio-ML Agent Deployment Rehberi

Bu doküman, Bio-ML Agent'ı farklı ortamlarda nasıl yayına alacağınızı açıklar.

## 🐳 Docker ve Docker Compose (Önerilen)

En hızlı ve tüm servisleri (Ajan + MLflow) içeren yöntemdir.

1. **Sistemi Başlat:**
   ```bash
   docker-compose up -d
   ```
2. **Erişim:**
   - **Agent API:** `http://localhost:8000`
   - **MLflow UI:** `http://localhost:5001`

## 🤗 HuggingFace Spaces Deployment

Bio-ML Agent'ı HuggingFace üzerinde bir "Space" olarak çalıştırmak için:

1. HF üzerinde yeni bir **Docker Space** oluşturun.
2. `Dockerfile` ve projenin tüm dosyalarını yükleyin.
3. `config.yaml` içindeki API anahtarlarını (OpenAI/Anthropic vb.) HF **Secrets** bölümüne ekleyin.
4. HF otomatik olarak imajı build edecek ve yayına alacaktır.

## ☁️ Railway / Render Deployment

1. GitHub deponuzu Railway'e bağlayın.
2. Railway otomatik olarak `Dockerfile`'ı algılayacaktır.
3. Gerekli ortam değişkenlerini (Variables) Railway panelinden tanımlayın.
4. **Volume:** Verilerin kalıcı olması için `/app/workspace` ve `/app/db` dizinlerini bir volume'e bağlamayı unutmayın.

## 🛠️ Manuel Kurulum (Development)

1. Depoyu klonlayın.
2. Bağımlılıkları kurun: `pip install -e ".[all]"`
3. API sunucusunu başlatın:
   ```bash
   uvicorn api_server:app --reload
   ```
