# Bio-ML Agent Deployment Guide

This document explains how to deploy Bio-ML Agent in different environments.

## 🐳 Docker & Docker Compose (Recommended)

The fastest method — starts all services together.

1. **Start the system:**
   ```bash
   docker-compose up -d
   ```
2. **Access:**

| Service | Port | Description |
| :--- | :--- | :--- |
| `redis` | 6380 | Message queue and cache (RQ / background jobs) |
| `api` | 8001 | FastAPI REST server + Platform API |
| `worker` | — | Background worker (RQ jobs) |
| `web_ui` | 7860 | Gradio web interface |
| `mlflow` | 5005 | MLflow Tracking Server |
| `litellm` | 4000 | Multi-LLM Proxy & Routing Gateway |
| `qdrant` | 6333 | Vector memory (RAG) |
| `postgres` | 5432 | PostgreSQL database (optional data storage) |
| `minio` | 9000 / 9001 | Object storage (artifacts) |
| `otel_collector` | 4317 / 4318 | OpenTelemetry collector |
| `gateway` | 8000 | External-facing API gateway |
| `notification_worker` | — | Background notification processor |

3. **Tail logs:**
   ```bash
   docker-compose logs -f worker
   docker-compose logs -f api
   ```

---

## 🤗 HuggingFace Spaces Deployment

To run Bio-ML Agent as a HuggingFace Space:

1. Create a new **Docker Space** on HuggingFace.
2. Upload the `Dockerfile` and all project files.
3. Add your API keys (OpenAI/Anthropic/Google etc.) to the HF **Secrets** section (`config.yaml` reads from environment variables).
4. HuggingFace will automatically build the image and deploy.

---

## ☁️ Railway / Render Deployment

1. Connect your GitHub repository to Railway.
2. Railway will automatically detect the `Dockerfile`.
3. Define the required environment variables in the Railway dashboard.
4. **Volume**: Mount `/app/workspace` and `/app/data` to a persistent volume so data survives restarts.

---

## 🛠️ Manual / Development Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -e ".[all]"
   ```
3. Start the API server:
   ```bash
   python run_api.py         # or: bio-ml-agent api
   ```
4. Start the Web UI (separate terminal):
   ```bash
   python run_ui.py          # or: bio-ml-agent ui
   ```
5. (Optional) Start WhatsApp connector:
   ```bash
   python run_whatsapp.py
   ```

---

## ⚡ Unified Launcher

`start_bio_ml.py` starts all processes (Web UI + WhatsApp connector + ngrok tunnel) in a single command:
```bash
python start_bio_ml.py
```
Requires a `.env` file with LLM and Twilio API keys.

---

## 🇹🇷 Türkçe Deployment Özeti (Turkish)

**Docker (Önerilen):**
```bash
docker-compose up -d
```
12 servis başlar. Temel erişim noktaları:
- Platform API: `http://localhost:8001`
- Web UI (Gradio): `http://localhost:7860`
- MLflow: `http://localhost:5005`
- Gateway: `http://localhost:8000`

**Manuel Kurulum:**
```bash
pip install -e ".[all]"
python run_api.py   # API sunucusu
python run_ui.py    # Web arayüzü
```
