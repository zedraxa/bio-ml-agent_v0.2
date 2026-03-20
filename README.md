<div align="center">

![Bio-ML Agent Banner](assets/banner.png)

# 🧬 Bio-ML Agent
**Biyomühendislik Odaklı, Otonom ve Açıklanabilir ML İş Akışı Asistanı**

[![Version](https://img.shields.io/badge/version-v0.1.0--clean-green.svg)](#-sürümleme)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://github.com/zedraxa/bio-ml-agent_v0.2/actions/workflows/test.yml/badge.svg)](https://github.com/zedraxa/bio-ml-agent_v0.2/actions)

</div>

---

## 🌟 Neden Bio-ML Agent? (Farkımız)

Piyasadaki genel amaçlı ajanların aksine, Bio-ML Agent **Biyomühendislik ve Yaşam Bilimleri** için özel olarak optimize edilmiştir:

-   **🧬 Domain-Specific Expertise:** Ajanlar; protein analizi, genomik veri işleme ve klinik veri setleri konusunda önceden tanımlı stratejilere sahiptir.
-   **🔒 Local-First & Privacy:** Hassas biyomedikal verileriniz için Ollama üzerinden %100 yerel modda çalışabilir.
-   **🧪 Explainability (XAI):** Sadece sonuç vermez; SHAP ve LIME entegrasyonu ile modelin neden bu tahmini yaptığını bilimsel olarak açıklar.
-   **🛠️ Sandbox Güvenliği:** Üretilen kodlar izole sandbox ortamında çalıştırılır, sistem güvenliğiniz riske atılmaz.

---

![Dashboard Preview](assets/dashboard.png)

---

## ⚡ Hızlı Başlangıç

```bash
# 1. Klonla ve Ayarları Yap
git clone https://github.com/zedraxa/bio-ml-agent_v0.2.git && cd bio-ml-agent_v0.2
cp .env.example .env

# 2. Docker ile Başlat (Tüm Servisler)
docker-compose up -d

# 3. Veya Yerel Olarak Başlat
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
python3 web_ui.py
```

---

## 🏆 Golden Path Demo

Sistemi en iyi şekilde deneyimlemek için hazır senaryomuzu kullanın:

1.  **Demo Verisini Üretin:**
    ```bash
    python examples/golden_path_demo/generate_demo_data.py
    ```
2.  **Senaryoyu Takip Edin:** `examples/golden_path_demo/demo_scenario.md` içindeki promptları ajana göndererek uçtan uca analiz yapın.

---

## 🏗️ Mimari Yapı

Bio-ML Agent, ölçeklenebilir ve güvenilir bir yapı için katmanlı mimari kullanır:
-   **Core:** Akıllı orkestrasyon ve tool yönetimi.
-   **Services:** `AgentService`, `RAGEngine` ve `JobWorker`.
-   **UI:** `Gradio` tabanlı modern web arayüzü ve WhatsApp entegrasyonu.

---

## 📌 Sürümleme

Bu proje **Semantic Versioning** prensiplerini takip eder. 
Güncel kararlı sürüm: `v0.1.0-clean`

-   `clean` suffix'i: Mimari sadeleştirme ve test süreçleri tamamlanmış, üretim öncesi hazır sürümü ifade eder.

---

## 👤 İletişim ve Geliştirici

**Yusuf Kavak** — [@zedraxa](https://github.com/zedraxa)

Proje ile ilgili sorularınız için GitHub Issues üzerinden ulaşabilirsiniz.
MIT Lisansı ile korunmaktadır.
