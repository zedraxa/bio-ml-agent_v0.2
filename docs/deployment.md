# Bio-ML Agent Deployment Guide

This document explains how to deploy Bio-ML Agent in different environments.

## 🐳 Docker & Docker Compose (Recommended)

The fastest method — starts all services (Agent + MLflow) together.

1. **Start the system:**
   ```bash
   docker-compose up -d
   ```
2. **Access:**
   - **Agent API:** `http://localhost:8000`
   - **MLflow UI:** `http://localhost:5001`

## 🤗 HuggingFace Spaces Deployment

To run Bio-ML Agent as a HuggingFace Space:

1. Create a new **Docker Space** on HuggingFace.
2. Upload the `Dockerfile` and all project files.
3. Add your API keys (OpenAI/Anthropic etc.) to the HF **Secrets** section in `config.yaml`.
4. HuggingFace will automatically build the image and deploy.

## ☁️ Railway / Render Deployment

1. Connect your GitHub repository to Railway.
2. Railway will automatically detect the `Dockerfile`.
3. Define the required environment variables in the Railway dashboard.
4. **Volume**: Mount `/app/workspace` and `/app/db` to a persistent volume so data survives restarts.

## 🛠️ Manual Installation (Development)

1. Clone the repository.
2. Install dependencies: `pip install -e ".[all]"`
3. Start the API server:
   ```bash
   uvicorn api_server:app --reload
   ```

---

## 🇹🇷 Türkçe Deployment Özeti (Turkish)

**Docker (Önerilen):**
```bash
docker-compose up -d
```
Agent API: `http://localhost:8000` | MLflow: `http://localhost:5001`

**HuggingFace Spaces:** Docker Space oluşturun, dosyaları yükleyin, API anahtarlarını Secrets'a ekleyin.

**Railway/Render:** GitHub deposunu bağlayın, Dockerfile otomatik algılanır; `/app/workspace` için kalıcı volume ekleyin.

**Manuel:** `pip install -e ".[all]"` ardından `uvicorn api_server:app --reload`
