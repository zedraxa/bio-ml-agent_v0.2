# REST API Referansı

Bio-ML Agent, `api_server.py` üzerinden modern bir RESTful API sunar.

## 🔗 Temel Endpointler

### 1. CNN Eğitimi Tetikleme
`POST /api/v1/agent/train_cnn`
- **Girdi:** `dataset_path`, `preset`, `architecture`, `epochs`
- **Yanıt:** `task_id` (Asenkron işlem)

### 2. RAG İndeksleme
`POST /api/v1/rag/index`
- **Girdi:** Yok (Tüm workspace'i tarar)
- **Yanıt:** `task_id`

### 3. Görev Durumu Sorgulama
`GET /api/v1/agent/status/{task_id}`
- **Yanıt:** Görevin durumu (`running`, `completed`, `error`) ve ajan raporu.

## 🔒 Güvenlik
API isteklerinde `X-API-Key` header'ı kullanılmalıdır. API anahtarı `config.yaml` içinden ayarlanabilir.
