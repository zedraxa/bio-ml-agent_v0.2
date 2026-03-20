# Capability Matrix (Mimari Göç Durumu)

Sistem şu anda büyük bir **Mimari Geçiş (Transition Architecture)** aşamasındadır. Eski monolitik yapıdan, çok katmanlı, izole ve modüler `ultra_agent` altyapısına geçilmektedir. Hangi modülün ne durumda olduğunu aşağıdaki tablodan görebilirsiniz:

| Bileşen / Özellik | Durum | Dizin / Modül | Açıklama |
| :--- | :--- | :--- | :--- |
| **API & Web UI** | 🟢 Production Ready | `api_server.py`, `web_ui.py` | FastAPI ve Gradio tabanlı ana kullanıcı çalışma alanı ve dış entegrasyon noktası. |
| **Asenkron Worker** | 🟢 Production Ready | `job_worker.py`, `services/` | Redis tabanlı kuyruk yönetimi ve arkaplan görev işleyicisi. Hata toleransı barındırır. |
| **Swarm Orchestrator** | 🟢 Production Ready | `swarm/` | Çoklu ajan (Veri Mühendisi, ML, Bio Uzman) koordinasyon motoru. |
| **Unified Storage Plane** | 🟢 Production Ready | `models/`, `services/` | Object Storage, Cache, Vector Store, Secret Vault, Experiment Registry 7-katmanlı veri servisleri. |
| **Unified Execution Plane** | 🟢 Production Ready | `models/`, `services/` | Graph yönetici, Handoff (WhatsApp/Web onayı), Hibrit Worker & Otonom Donanım Seçici. |
| **Unified Observability** | 🟢 Production Ready | `models/`, `services/` | Tarayıcı hata kaydı, run timeline, span, per-user audit (denetim) ve LLM cloud bütçe faturalaması. |
| **Gelişmiş Hafıza (v2)** | 🟡 Beta (Active) | `ultra_agent/memory/` | Qdrant tabanlı, LLM değerlendirmeli (Precision, Hit Rate), otomatik sentezleme (Merge) yapabilen yeni nesil anlamsal bellek. |
| **Browser Agent / DOM** | 🟡 Beta (Active) | `ultra_agent/runtime/` | DOM-Intelligent yapıya sahip, izole profillerde çalışan ve hata ayıklama (P7-Trace) sunan Tarayıcı otonomisi. |
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
