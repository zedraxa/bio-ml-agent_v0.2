<div align="center">

# 🧠 Bio-ML Agent

**Biyomühendislik ve Makine Öğrenimi Proje Asistanı**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/Tests-329%2B%20passed-brightgreen.svg)](#-testler)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![LLM](https://img.shields.io/badge/LLM-Gemini%20|%20OpenAI%20|%20Ollama-purple.svg)](#-desteklenen-llm-backendleri)

<p align="center">
Doğal dil komutlarıyla tam kapsamlı ML projeleri oluşturan otonom bir yapay zeka ajanı.<br>
Veri seti indirme → Model eğitimi → Etkileşimli Görselleştirme → RAG Bellek → Rapor oluşturma<br>
<i>Whatsapp ve Gradio (Ses & Görüntü) üzerinden kesintisiz erişim!</i>
</p>

</div>

---

## 🎯 Ne Yapar?

Bir cümle yazarsınız, agent sizin için **komple bir ML projesi** oluşturur:

```
>>> PROJECT: diabetes Breast Cancer veri setini kullanarak sınıflandırma modeli oluştur.
    En az 3 model karşılaştır, grafikleri ve raporu oluştur.
```

**Sonuç:**
- ✅ Veri seti indirilir (`data/raw/`)
- ✅ 5 farklı model eğitilir ve karşılaştırılır (5-fold CV)
- ✅ 6 analiz grafiği oluşturulur (confusion matrix, ROC curve, vb.)
- ✅ JSON sonuç dosyası + Markdown karşılaştırma raporu
- ✅ Detaylı Türkçe `report.md` ve `README.md`

---

## ⚡ Hızlı Başlangıç

### Kurulum

```bash
# 1. Repo'yu klonla
git clone https://github.com/zedraxa/bio-ml-agent_v0.2.git
cd bio-ml-agent_v0.2

# 2. Sanal ortam oluştur
python3 -m venv venv
source venv/bin/activate

# 3. Bağımlılıkları kur (Örn: Hem Web UI hem Bulut LLM destekli)
pip install -e ".[cloud,ui]"

# Sadece minimal çekirdek asistan için:
pip install -e .
```

### Çalıştırma

#### Gemini ile (Önerilen)
```bash
export GEMINI_API_KEY="YOUR_API_KEY"
python3 agent.py --model gemini-2.5-flash
```

#### Ollama ile (Yerel & Ücretsiz)
```bash
# Önce Ollama kur: https://ollama.ai
ollama pull qwen2.5:7b-instruct
python3 agent.py --model qwen2.5:7b-instruct --backend local
```

#### OpenAI ile
```bash
export OPENAI_API_KEY="YOUR_API_KEY"
python3 agent.py --model gpt-4o --backend remote
```

---

## 🤖 Desteklenen LLM Backend'leri

| Backend | API | Komut |
|---------|-----|-------|
| **Google Gemini** | `google-genai` | `--model gemini-2.5-flash` |
| **Ollama** (Yerel) | Yerel API | `--model qwen2.5:7b-instruct --backend local` |
| **OpenAI** | OpenAI API | `--model gpt-4o --backend remote` |
| **Anthropic** | Claude API | `--model claude-3-5-sonnet-20241022 --backend remote` |

> Model adına göre otomatik backend seçimi yapılır (`auto` mod).

---

## 📊 Örnek Çıktı

Agent ile oluşturulmuş bir Breast Cancer sınıflandırma projesi:

```
workspace/diabetes/
├── data/raw/                         # Veri seti
├── src/train.py                      # Eğitim kodu
├── utils/
│   ├── model_compare.py              # Çoklu model karşılaştırma
│   └── visualize.py                  # Görselleştirme araçları
├── results/
│   ├── plots/
│   │   ├── confusion_matrix.png      # Karmaşıklık Matrisi
│   │   ├── roc_curve.png             # ROC Eğrisi
│   │   ├── feature_importance.png    # Özellik Önemi
│   │   ├── correlation_matrix.png    # Korelasyon Matrisi
│   │   ├── learning_curve.png        # Öğrenme Eğrisi
│   │   └── class_distribution.png    # Sınıf Dağılımı
│   ├── comparison_results.json       # Model metrikleri
│   └── comparison_report.md          # Karşılaştırma raporu
├── report.md                         # Detaylı proje raporu
├── README.md                         # Proje açıklaması
└── pyproject.toml                    # Proje ve Bağımlılıklar
```

### Model Karşılaştırma Sonuçları

| Model | Accuracy | F1 Score | ROC AUC |
|-------|----------|----------|---------|
| **Logistic Regression** 🏆 | %98.2 | %98.6 | %99.6 |
| SVM | %98.2 | %98.6 | %99.5 |
| Random Forest | %95.6 | %96.6 | %99.4 |
| Gradient Boosting | %95.6 | %96.6 | %99.1 |
| KNN | %95.6 | %96.6 | %97.9 |

---

## 🛠️ Özellikler

### Çekirdek Özellikler
- 🧠 **Çoklu LLM Desteği** — 4 farklı backend (Gemini, OpenAI, Anthropic, Ollama)
- 📊 **Otomatik Model Karşılaştırma** — 5+ model, 5-fold cross-validation, metrik tablosu
- 📈 **Etkileşimli Görselleştirme (Plotly)** — Statik grafikler yerine yakınlaştırılabilir HTML tabanlı dinamik arayüz (ROC, Confusion Matrix vb.)
- 📝 **Otomatik Rapor** — Türkçe markdown rapor + README oluşturma
- 🔒 **Güvenlik** — Tehlikeli komut engelleme, path traversal koruması, timeout

### İleri Düzey Yetenekler (V5)
- 📱 **WhatsApp Bot Entegrasyonu** — Uzaktan mesajlaşarak (Örn: "Diyabet verisiyle model eğit") ML projeleri üretebilme
- 🧠 **Uzun Dönem Hafıza (RAG)** — ChromaDB Vectordb tabanlı bellek ile eski projeleri ve sohbetleri hatırlama
- 🎙️ **Sesli Etkileşim (Voice)** — Gradio UI üzerinden mikrofon komutlarıyla veri analizi yapma
- 👁️ **Görüntü İşleme (Vision)** — Tıbbi görüntüleri (MRI vb.) veya grafik verilerini okuyarak hastalık tahmini ve analizi yapabilme

### ML Araçları & Altyapı
- 📂 **15+ Yerleşik Veri Seti** — breast_cancer, diabetes, wine_quality, heart_disease, iris...
- 🔬 **Biyomühendislik Toolkit** — Protein analizi (PDB İndirme), genomik, atık su, ilaç molekülü (Lipinski)
- 🔌 **Plugin Sistemi** — Özel tool'lar ekleyerek genişletilebilir
- 🌐 **Data Explorer (Gradio)** — Anlık analiz sonuçlarını sekme üzerinden direkt görüntüleme
- ✅ **329+ Unit Test** — Yüksek test coverage ve kararlı mimari

### Altyapı
- 💬 **Konuşma Geçmişi** — Oturumları kaydet, yükle, devam ettir
- 📋 **Merkezi Yapılandırma** — `config.yaml` ile tüm ayarları kontrol et
- 🌐 **Web Arayüzü** — Gradio tabanlı chat UI (`web_ui.py`)

---

## 📁 Proje Yapısı

```
bio-ml-agent_v0.2/
├── agent.py                 # Ana agent (1092 satır)
├── llm_backend.py           # Çoklu LLM backend (425 satır)
├── exceptions.py            # 7 özel hata sınıfı
├── bioeng_toolkit.py        # Biyomühendislik araçları
├── dataset_catalog.py       # 15+ veri seti kataloğu
├── report_generator.py      # Otomatik rapor oluşturucu
├── mlflow_tracker.py        # MLflow entegrasyonu
├── plugin_manager.py        # Plugin sistemi
├── web_ui.py                # Gradio web arayüzü
├── progress.py              # Terminal spinner
├── config.example.yaml      # Merkezi yapılandırma şablonu (Kopyalayıp config.yaml yapın)
├── utils/
│   ├── config.py            # YAML + env yapılandırma
│   ├── model_compare.py     # Çoklu model karşılaştırma
│   └── visualize.py         # ML görselleştirme
├── plugins/                 # Özel plugin'ler
├── swarm/                   # Çoklu ajan mimarisi (V6)
├── tests/                   # 329+ unit test
└── workspace/               # Agent çıktıları
```

---

## 🧪 Testler

```bash
# Tüm testleri çalıştır
python -m pytest tests/ -x -q

# Sonuç:
# 329 passed in 12.25s ✅
```

---

## ⌨️ Agent Komutları

| Komut | Açıklama |
|-------|----------|
| `exit` / `quit` | Agent'tan çık |
| `/history` | Kayıtlı oturumları listele |
| `/load <id>` | Önceki oturumu yükle |
| `/save` | Mevcut oturumu kaydet |
| `/new` | Yeni oturum başlat |
| `/delete <id>` | Oturumu sil |
| `/info` | Oturum bilgilerini göster |
| `/logs [n]` | Son n log satırını göster |

---

## 🔧 Yapılandırma

`config.yaml` dosyasıyla tüm ayarları kontrol edin:

```yaml
workspace:
  base_dir: workspace

agent:
  model: gemini-2.5-flash
  max_steps: 30
  timeout: 180

security:
  allow_web_search: false
  deny_patterns:
    - '\brm\b.*-rf\s+/'
    - '\bshutdown\b'
```

---

## 📄 Dokümantasyon

- 📖 [Kullanma Kılavuzu](KULLANMA_KILAVUZU.md) — Detaylı kullanım rehberi
- 📊 [Proje Raporu](RAPOR.md) — Kapsamlı teknik rapor (v3)

---

## 🗺️ Başarılan Yol Haritası

- [x] Çoklu LLM backend desteği (Gemini, OpenAI, Ollama vs.)
- [x] Özel Biyomühendislik Araçları (Bioeng Toolkit)
- [x] Otomatik ML modeli eğitme ve raporlama
- [x] Konuşma geçmişi (RAG) & Uzun dönem vektör veritabanı belleği
- [x] Web arayüzü (Gradio) ve canlı yayın (Streaming)
- [x] WhatsApp Bot Entegrasyonu ile Uzaktan ML Model Yönetimi
- [x] Ses (Voice) ve Görüntü (Vision) İşleme Entegrasyonları
- [x] İnteraktif Plotly Görselleştirmeleri & Data Explorer Paneli
- [x] Docker desteği & CI/CD pipeline
- [x] Kapsamlı Test Kapsamı (329 Unit/E2E Test)

---

## 👤 Geliştirici

**Yusuf Kavak** — [@zedraxa](https://github.com/zedraxa)

---

<div align="center">

**⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!**

</div>
