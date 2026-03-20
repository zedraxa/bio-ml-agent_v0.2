# Capability Matrix (Mimari Göç Durumu)

Sistem şu anda büyük bir **Mimari Geçiş (Transition Architecture)** aşamasındadır. Eski monolitik yapıdan, çok katmanlı, izole ve modüler `ultra_agent` altyapısına geçilmektedir. Hangi modülün ne durumda olduğunu aşağıdaki tablodan görebilirsiniz:

| Bileşen / Özellik | Durum | Dizin / Modül | Açıklama |
| :--- | :--- | :--- | :--- |
| **API & Web UI** | 🟢 Production Ready | `api/`, `ui/web_ui.py` | FastAPI ve Gradio tabanlı ana kullanıcı çalışma alanı. |
| **Agent Service** | 🟢 Production Ready | `services/agent/` | Modularized orchestration, memory, and lifecycle management. |
| **Asenkron Worker** | 🟢 Production Ready | `workers/job_worker.py` | Redis tabanlı kuyruk yönetimi ve arkaplan görev işleyicisi. |
| **Swarm Orchestrator** | 🟢 Production Ready | `swarm/` | Çoklu ajan (Veri Mühendisi, ML, Bio Uzman) koordinasyon motoru. |
| **Unified Storage Plane** | 🟢 Production Ready | `models/`, `services/` | Object Storage, Cache, Vector Store, Secret Vault. |
| **Unified Execution Plane** | 🟢 Production Ready | `core/`, `agents/` | Graph yönetici, Handoff, Hibrit Worker. |
| **Unified Observability** | 🟢 Production Ready | `ultra_agent/metrics/` | Span, per-user audit ve LLM bütçe takibi. |
| **Gelişmiş Hafıza (v2)** | 🟡 Beta (Active) | `ultra_agent/memory/` | Qdrant tabanlı, otomatik sentezleme yapabilen yeni nesil bellek. |
| **Browser Agent / DOM** | 🟡 Beta (Active) | `ultra_agent/runtime/` | DOM-Intelligent Tarayıcı otonomisi. |
| **Eski Nesil RAG** | 🔴 Deprecated | `rag_engine.py` | Sistem kademeli olarak `ultra_agent/memory` altyapısına geçmiştir. |
| **Eski Nesil Agent** | 🔴 Deprecated | `agent.py` | Eski orkestratörler. Uyumluluk için kök dizinde kalmaya devam etmektedir. |

## 📂 Klasör Yapısı (Özet)

```
bio-ml-agent/
├── 🟢 ANA ÇALIŞMA YOLU (Production Path)
│   ├── web_ui.py                 # (UI) Kullanıcı girişi ve XAI paneli
│   ├── api_server.py             # (API) Dış istemciler ve Webhooklar
│   ├── job_worker.py             # (Worker) Görev kuyruk yönetimi
│   ├── services/                 # AgentService gibi temel servis bağlayıcıları
│   └── swarm/                    # (Orchestration) Çoklu-ajan hiyerarşisi
│
├── 🟡 DENEYSEL VE İLERİ YETENEKLER (Ultra Agent Hattı)
│   └── ultra_agent/              
│       ├── memory/               # Faz 9: Akıllı Hafıza (Merge, TTL, Qdrant)
│       ├── runtime/browser/      # Faz 16: İzole & DOM Zekalı Tarayıcı Ajanı
│       └── metrics/              # OTel & Prometheus Observability
│
└── 🔴 DEPRECATED (Geri kalmış, yavaşça silinecek dosyalar)
    ├── agent.py, multi_agent.py
    └── rag_engine.py, memory_manager.py
```
