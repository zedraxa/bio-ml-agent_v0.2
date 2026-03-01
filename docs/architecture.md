# Mimari Genel Bakış

Bio-ML Agent, servis odaklı ve olay-güdümlü (event-driven) bir mimari üzerine inşa edilmiştir.

## 🏗 Katmanlar

### 1. AgentService (Çekirdek Servis)
Tüm iş mantığının kalbi burasıdır. Gradio, CLI ve API bu servisi kullanarak ajanı tetikler. Durum güncellemelerini asenkron olarak döner.

### 2. LLM Backend (Çoklu Model Desteği)
OpenAI, Anthropic, Gemini ve Ollama gibi farklı sağlayıcıları standart bir arayüzle yönetir. Model yeteneklerini (vision, tools) otomatik olarak algılar.

### 3. RAG Engine (Doküman Zekası)
ChromaDB vektör veritabanı ve BM25 hibrit arama algoritmasını kullanarak yerel dokümanları indeksler ve sorgular.

### 4. Background Workers (Redis/RQ)
Zaman alan işlemler (ML eğitimi, derin doküman indeksleme) arka planda Redis kuyruğu ile yönetilir.

## 🔄 İş Akışı
1. Kullanıcıdan mesaj gelir.
2. `AgentService` mesajı alır, geçmişle birleştirir.
3. Uygun LLM modeli seçilir (`Capability Registry`).
4. Gerekirse RAG motorundan bağlam çekilir.
5. Model çıktı üretir, araçları (Python, Bash vb.) kullanır.
6. Sonuç kullanıcıya iletilir.
