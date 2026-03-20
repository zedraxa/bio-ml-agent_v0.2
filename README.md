<div align="center">

# 🧠 Bio-ML Agent
**Bio-insanlı Geliştirme ve Makine Öğrenmesi İş Akışları İçin Modüler Yapay Zeka Asistanı**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![LLM](https://img.shields.io/badge/LLM-Gemini%20|%20OpenAI%20|%20Ollama-purple.svg)](#-desteklenen-llm-backendleri)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](#-docker-ile-çalıştırma)

</div>

---

## 🎯 Proje Amacı
Bio-ML Agent, biyomühendislik ve veri bilimi projelerinde karmaşık iş akışlarını (veri temizleme, model eğitimi, açıklanabilirlik analizi, bilimsel raporlama) doğal dil komutlarıyla yöneten bir otonom sistemdir.

### Ana Özellikler:
- **Çoklu Uzman Ajan:** Veri Mühendisi, ML Uzmanı ve Biyoinformatik Uzmanı koordineli çalışır.
- **Deep Research:** İnternet üzerindeki bilimsel kaynakları tarayarak derinlemesine rapor hazırlar.
- **Explainable AI (XAI):** SHAP/LIME entegrasyonu ile modellerin kararlarını açıklar.
- **Sandbox Execution:** Üretilen kodları güvenli bir izole ortamda çalıştırır.

---

## ⚡ Hızlı Başlangıç (Tek Komutla)

```bash
# 1. Klonla ve Ayarları Yap
git clone https://github.com/zedraxa/bio-ml-agent_v0.2.git && cd bio-ml-agent_v0.2
cp .env.example .env  # API anahtarlarınızı buraya ekleyin

# 2. Docker ile Tüm Sistemi Başlat (Önerilen)
docker-compose up -d

# 3. Veya Yerel Olarak Başlat
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
python3 web_ui.py
```

---

## 🏗️ Mimari Özet
Sistem üç ana katmandan oluşur:
1. **Core (Ajan Çekirdeği):** Prompt yönetimi, tool execution ve LLM yönlendirme.
2. **Services (Servis Katmanı):** Proje yönetimi, RAG bellek ve background görevler.
3. **Interfaces (Arayüzler):** Web UI (Gradio), API (FastAPI) ve WhatsApp Gateway.

---

## 🕹️ Modlar ve Yetenekler

| Mod | Açıklama | Ne Zaman Kullanılır? |
|-----|-----------|--------------------|
| **Chat Mode** | Genel sorular ve hızlı veri inceleme. | Basit analizler ve etkileşimli yardım. |
| **Deep Research** | İnternet taraması ve sentezleme. | Bilimsel literatür taraması ve derin raporlama. |
| **Project Mode** | Uçtan uca ML boru hattı oluşturma. | Veri setinden çalışan bir modele gitmek için. |
| **XAI Mode** | Model kararlarının görselleştirilmesi. | Modelin neden "hasta" dediğini anlamak için. |

---

## 📚 Dokümantasyon
Detaylı bilgi için dokümantasyon merkezini ziyaret edin:
👉 **[Bio-ML Agent Dokümantasyon Portalı](docs/index.md)**

---

## 👤 Geliştirici
**Yusuf Kavak** — [@zedraxa](https://github.com/zedraxa)
