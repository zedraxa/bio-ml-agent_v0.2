# REST API Reference

Bio-ML Agent exposes a modern RESTful API via `api_server.py`.

## 🔗 Core Endpoints

### 1. Trigger CNN Training
`POST /api/v1/agent/train_cnn`
- **Input**: `dataset_path`, `preset`, `architecture`, `epochs`
- **Response**: `task_id` (asynchronous operation)

### 2. RAG Indexing
`POST /api/v1/rag/index`
- **Input**: None (scans the entire workspace)
- **Response**: `task_id`

### 3. Query Task Status
`GET /api/v1/agent/status/{task_id}`
- **Response**: Task state (`running`, `completed`, `error`) and the agent report.

## 🔒 Security
All API requests must include an `X-API-Key` header. The API key is configured in `config.yaml`.

---

## 🇹🇷 Türkçe API Özeti (Turkish)

Bio-ML Agent, `api_server.py` üzerinden RESTful API sunar.

- `POST /api/v1/agent/train_cnn` — CNN eğitimi başlatır, `task_id` döner.
- `POST /api/v1/rag/index` — Workspace'i tarayıp RAG indeksler.
- `GET /api/v1/agent/status/{task_id}` — Görev durumunu sorgular.

Güvenlik: İsteklerde `X-API-Key` header'ı gereklidir; anahtar `config.yaml`'da ayarlanır.
