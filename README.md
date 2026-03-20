<div align="center">

# 🧠 Bio-ML Agent

**Modular AI assistant for bioengineering and ML workflows**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![LLM](https://img.shields.io/badge/LLM-Gemini%20|%20OpenAI%20|%20Ollama-purple.svg)](#-desteklenen-llm-backendleri)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](#-docker-ile-çalıştırma)

</div>

---

## 🎯 Ne Yapar?

Doğal dil komutuyla **3 uzman ajan** koordineli çalışarak komple ML projesi oluşturur:

```
>>> Buradaki data/raw/diabetes.csv verisini oku, temizle, model kur ve SHAP analizi yap.
```

**→ Veri Mühendisi** temizler → **ML Uzmanı** eğitir + SHAP üretir → **Biyoinformatik Uzman** klinik yorumlar

---

## ⚡ Kurulum ve Çalıştırma

```bash
# 1. Klonla
git clone https://github.com/zedraxa/bio-ml-agent_v0.2.git && cd bio-ml-agent_v0.2

# 2. Sanal ortam kur
python3 -m venv venv && source venv/bin/activate

# 3. Bağımlılıkları yükle
pip install -e ".[all]"

# 4. API Key ayarla
export GEMINI_API_KEY="YOUR_KEY"

# 5. Web arayüzünü başlat
python3 web_ui.py
# → Tarayıcıda http://localhost:7860
```

### 🐳 Docker ile Çalıştırma

```bash
docker-compose up -d
# API Server:  http://localhost:8001
# Web UI:      http://localhost:## 📚 Tam Dokümantasyon (Single Source of Truth)

Bio-ML Agent hakkındaki **tüm detaylı bilgilere, kullanım kılavuzuna, mimari detaylara ve API referanslarına** aşağıdaki bağlantıdan veya `docs/` klasöründen ulaşabilirsiniz:

👉 **[Bio-ML Agent Merkezi Dokümantasyon Portalı](docs/index.md)** (veya `mkdocs serve` ile yerel olarak görüntüleyin)

### Başlıca Dokümantasyon Başlıkları:
- [Kullanım Kılavuzu](docs/kullanim_kilavuzu.md)
- [Capability Matrix & Mimari Durumu](docs/capability_matrix.md)
- [Mimari Genel Bakış](docs/architecture.md)
- [Katkıda Bulunma Rehberi](docs/contributing_guide.md)

---

## 👤 Geliştirici

**Yusuf Kavak** — [@zedraxa](https://github.com/zedraxa)

---

<div align="center">

**⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!**

</div>n Yol Haritası](ULTRA_AJAN_YOL_HARITASI.md) | Gelişmiş mimari planı ve tamamlanan görevler |

---

## ⚠️ Güvenlik & Denetim (Audit)
> **Uyarı:** Plugin sistemi allowlist (izin) bazlıdır. Sistemdeki Python kodu `SandboxRuntime` kısıtları (%50 CPU, memory lock) içerisinde çalışır.
> Bütün kritik operasyonlar `audit_logs/` klasörüne zaman damgasıyla değiştirilemez formatta yazılır.

---

## 👤 Geliştirici

**Yusuf Kavak** — [@zedraxa](https://github.com/zedraxa)

---

<div align="center">

**⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!**

</div>
