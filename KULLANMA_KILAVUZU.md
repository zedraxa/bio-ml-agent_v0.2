# 📖 Bio-ML Agent — Kullanma Kılavuzu

> **Sürüm:** 3.0  
> **Tarih:** 22 Şubat 2026  
> **Python:** 3.11+  
> **İşletim Sistemi:** Linux

---

## 📋 İçindekiler

1. [Kurulum](#1--kurulum)
2. [Hızlı Başlangıç](#2--hızlı-başlangıç)
3. [Terminal Arayüzü (CLI)](#3--terminal-arayüzü-cli)
4. [Web Arayüzü (Gradio)](#4--web-arayüzü-gradio)
5. [Görev Panosu (Dashboard)](#5--görev-panosu-dashboard)
6. [Yapılandırma](#6--yapılandırma)
7. [Agent Komutları](#7--agent-komutları)
8. [Tool Sistemi](#8--tool-sistemi)
9. [ML Proje Oluşturma](#9--ml-proje-oluşturma)
10. [Biyomühendislik Araçları](#10--biyomühendislik-araçları)
11. [Plugin Sistemi](#11--plugin-sistemi)
12. [LLM Backend Değiştirme](#12--llm-backend-değiştirme)
13. [Konuşma Geçmişi](#13--konuşma-geçmişi)
14. [Sorun Giderme](#14--sorun-giderme)
15. [Komut Referansı](#15--komut-referansı)

---

## 1. 🔧 Kurulum

### Ön Gereksinimler

- Python 3.11 veya üstü
- [Ollama](https://ollama.ai/) (yerel LLM için)
- pip (Python paket yöneticisi)

### Adım Adım Kurulum

```bash
# 1. Proje dizinine gidin
cd /home/yusuf/ai-agent\ \(diğer\ kopya\)/

# 2. Sanal ortamı oluşturun (zaten varsa bu adımı atlayın)
python3 -m venv venv

# 3. Sanal ortamı aktifleştirin
source venv/bin/activate

# 4. Bağımlılıkları yükleyin
pip install -r requirements.txt

# 5. Ollama modelini indirin (Ollama çalışır durumda olmalı)
ollama pull qwen2.5:7b-instruct
```

> **⚠️ Önemli Not:** Proje klasörü kopyalandıysa veya taşındıysa venv bozulabilir.
> Bu durumda eski venv'i silip yeniden oluşturun:
> ```bash
> rm -rf venv
> python3 -m venv venv
> source venv/bin/activate
> pip install -r requirements.txt
> ```

### Bağımlılıklar

| Paket | Amaç |
|-------|------|
| `ollama` | Yerel LLM API |
| `pyyaml` | Yapılandırma dosyası |
| `requests` | Web istekleri |
| `beautifulsoup4` | HTML parse |
| `duckduckgo-search` | Web araması |
| `scikit-learn` | ML modelleri |
| `pandas` | Veri işleme |
| `numpy` | Sayısal hesaplama |
| `matplotlib` | Grafikler |
| `seaborn` | İstatistik grafikleri |
| `pytest` | Testler |
| `flask` | Dashboard web sunucusu |
| `gradio` | Gradio web arayüzü |

---

## 2. 🚀 Hızlı Başlangıç

### Terminal Modunda Çalıştırma

```bash
# Sanal ortamı aktifleştirin
source venv/bin/activate

# Agent'ı başlatın
python3 agent.py
```

Agent başladığında şunu göreceksiniz:

```
🧠 Bio-ML Agent ready | model=qwen2.5:7b-instruct | workspace=/path/to/workspace
📜 Oturum ID: 20260222_014400_a1b2c3d4
💾 Geçmiş klasörü: /path/to/conversation_history
📋 Log klasörü: /path/to/logs
Çıkmak için: exit / quit | Komutlar: /history /load /new /save /delete /info /logs
```

### İlk Projenizi Oluşturun

```
>>> PROJECT: breast_cancer Meme kanseri veri setini kullanarak bir sınıflandırma modeli oluştur
```

Agent otomatik olarak:
1. Veri setini yükler
2. Proje yapısını oluşturur
3. Modelleri eğitir ve karşılaştırır
4. Grafikleri oluşturur
5. Rapor yazar

---

## 3. 💻 Terminal Arayüzü (CLI)

### Başlatma Seçenekleri

```bash
# Varsayılan ayarlarla başlat
python3 agent.py

# Farklı model kullan
python3 agent.py --model llama3:latest

# Özel çalışma alanı
python3 agent.py --workspace /tmp/my_workspace

# Timeout süresini artır
python3 agent.py --timeout 300

# Debug modunda logla
python3 agent.py --log-level DEBUG

# Var olan bir oturumu yükle
python3 agent.py --load-session 20260220_150000_abcd1234

# Özel config dosyası
python3 agent.py --config /path/to/custom_config.yaml
```

### CLI Argüman Tablosu

| Argüman | Varsayılan | Açıklama |
|---------|------------|----------|
| `--model` | `qwen2.5:7b-instruct` | Ollama model adı |
| `--workspace` | `workspace` | Çalışma alanı klasörü |
| `--timeout` | `180` | Komut timeout (saniye) |
| `--max-steps` | `50` | Maks. tool adımı |
| `--history-dir` | `conversation_history` | Geçmiş kayıt klasörü |
| `--load-session` | - | Başlangıçta yüklenecek oturum |
| `--log-level` | `INFO` | Log seviyesi |
| `--log-dir` | `logs` | Log klasörü |
| `--config` | `config.yaml` | Config dosya yolu |

---

## 4. 🌐 Web Arayüzü (Gradio)

### Başlatma

```bash
source venv/bin/activate
python3 web_ui.py
```

Tarayıcınızda `http://localhost:7860` adresini açın.

### Özellikler

- Chat kutusu ile agent ile sohbet
- Model, timeout, max_steps ayarlarını arayüzden değiştirme
- Yeni oturum başlatma
- Oturum listesini görüntüleme

---

## 5. 📊 Görev Panosu (Dashboard)

Proje, tüm modülleri ve görevleri tek bir yerden yönetebileceğiniz **Flask tabanlı bir web panosu** içerir.

### Başlatma

```bash
source venv/bin/activate
python3 dashboard.py
```

Tarayıcınızda `http://localhost:5050` adresini açın.

### Dashboard Özellikleri

| Özellik | Açıklama |
|---------|----------|
| **Görev Yönetimi** | Görev oluşturma, düzenleme, silme, onaylama ve reddetme |
| **Proje İstatistikleri** | Toplam kod satırı, modül sayısı, test sayısı |
| **Modül Bilgileri** | Her modülün satır sayısı ve boyutu |
| **Rapor Görüntüleme** | `RAPOR.md` dosyasının içeriğini doğrudan panoda okuma |
| **Yapılandırma Yönetimi** | `config.yaml` dosyasını arayüzden görüntüleme ve düzenleme |
| **API Key Yönetimi** | OpenAI, Anthropic, Google, HuggingFace API key'lerini güvenli ayarlama |
| **Ollama Model Listesi** | Yerel Ollama sunucusundaki modelleri listeleme |
| **Agent Chat** | Pano üzerinden doğrudan agent ile sohbet etme |

### Dashboard API Endpointleri

| Endpoint | Metod | Açıklama |
|----------|-------|----------|
| `GET /` | GET | Dashboard ana sayfası |
| `GET /api/tasks` | GET | Tüm görevleri getir (?status= filtresi) |
| `POST /api/tasks` | POST | Yeni görev oluştur |
| `PUT /api/tasks/<id>` | PUT | Görevi güncelle |
| `DELETE /api/tasks/<id>` | DELETE | Görevi sil |
| `POST /api/tasks/<id>/approve` | POST | Görevi onayla |
| `POST /api/tasks/<id>/reject` | POST | Görevi reddet |
| `GET /api/stats` | GET | Proje istatistikleri |
| `GET /api/report` | GET | RAPOR.md içeriği |
| `GET /api/modules` | GET | Modül bilgileri |
| `GET /api/config` | GET | Yapılandırmayı getir |
| `PUT /api/config` | PUT | Yapılandırmayı güncelle |
| `GET /api/api-keys` | GET | API key durumlarını getir |
| `POST /api/api-keys` | POST | API key'leri kaydet |
| `GET /api/ollama-models` | GET | Ollama modellerini listele |
| `POST /api/agent/chat` | POST | Agent'a mesaj gönder |

---

## 6. ⚙️ Yapılandırma

### config.yaml

Tüm ayarlar `config.yaml` dosyasından yönetilir:

```yaml
# Agent Ayarları
agent:
  model: "qwen2.5:7b-instruct"
  max_steps: 50
  timeout: 180
  language: "tr"

# Güvenlik
security:
  allow_web_search: false
  deny_patterns:
    - '\brm\b.*-rf\s+/'
    - '\bshutdown\b'
    - '\breboot\b'

# Çalışma Alanı
workspace:
  default_project: "scratch_project"
  base_dir: "workspace"
  auto_save_web: true

# Konuşma Geçmişi
history:
  directory: "conversation_history"
  auto_save_interval: 5

# Loglama
logging:
  level: "INFO"
  directory: "logs"
  file_name: "agent.log"
  max_bytes: 5242880        # 5 MB
  backup_count: 3
  console_level: "WARNING"

# ML Ayarları
ml:
  test_size: 0.2
  random_state: 42
  cv_folds: 5
  default_task: "classification"
  comparison:
    enabled: true
    generate_plots: true
    plot_dpi: 150
    output_formats:
      - json
      - csv
      - markdown
```

### Yapılandırma Öncelik Sırası

```
CLI argümanları > Ortam değişkenleri > config.yaml > Varsayılanlar
```

### Ortam Değişkenleri

```bash
export AGENT_MODEL="llama3:latest"
export AGENT_TIMEOUT=300
export OLLAMA_HOST="http://localhost:11434"
export OPENAI_API_KEY="sk-..."     # OpenAI backend için
export ANTHROPIC_API_KEY="sk-..."  # Anthropic backend için
export GOOGLE_API_KEY="..."        # Google Gemini backend için
export HF_API_TOKEN="hf_..."      # HuggingFace backend için
```

---

## 7. 📝 Agent Komutları

### Oturum Yönetimi

| Komut | Açıklama |
|-------|----------|
| `/history` | Kayıtlı oturumları listele |
| `/load <session_id>` | Eski bir oturumu yükle |
| `/new` | Yeni oturum başlat (mevcut kaydedilir) |
| `/save` | Mevcut oturumu hemen kaydet |
| `/delete <session_id>` | Bir oturumu sil |
| `/info` | Mevcut oturum bilgilerini göster |
| `/logs [N]` | Son N log satırını göster (varsayılan: 30) |
| `/help` veya `/h` | Yardım menüsü |
| `exit` veya `quit` | Çıkış (oturum kaydedilir) |

### Proje Belirtme

Mesajınıza `PROJECT: proje_adı` ekleyerek bir proje adı belirleyebilirsiniz:

```
>>> PROJECT: su_kalitesi Su kalitesi tahmin modeli oluştur
```

Belirtilmezse `scratch_project` kullanılır.

### Web Araması

Web araması varsayılan olarak **kapalıdır**. Etkinleştirmek için mesajınıza `ALLOW_WEB_SEARCH` ekleyin:

```
>>> ALLOW_WEB_SEARCH biyomühendislik veri setleri araştır
```

---

## 8. 🔧 Tool Sistemi

Agent, LLM'in çıktısındaki özel tag'leri algılayarak araçları çalıştırır.

### Dahili Tool'lar

| Tool | Tag | Açıklama |
|------|-----|----------|
| Python | `<PYTHON>...</PYTHON>` | Python kodu çalıştır |
| Bash | `<BASH>...</BASH>` | Bash komutu çalıştır |
| Web Search | `<WEB_SEARCH>...</WEB_SEARCH>` | DuckDuckGo araması |
| Web Open | `<WEB_OPEN>...</WEB_OPEN>` | URL'den metin çek |
| Read File | `<READ_FILE>...</READ_FILE>` | Dosya oku |
| Write File | `<WRITE_FILE>...</WRITE_FILE>` | Dosya yaz |
| TODO | `<TODO>...</TODO>` | Yapılacaklar listesi |

### WRITE_FILE Formatı

```
<WRITE_FILE>
path: proje/dosya.py
---
dosya içeriği buraya...
</WRITE_FILE>
```

---

## 9. 📊 ML Proje Oluşturma

### Desteklenen Veri Setleri

Agent dahili katalogunda **15+ veri seti** bulundurur:

| Veri Seti | Tür | Kategori |
|-----------|-----|----------|
| Breast Cancer | Binary Sınıflandırma | Medikal |
| Wine Quality | Multi Sınıflandırma | Genel |
| Diabetes | Regresyon | Medikal |
| Heart Disease | Binary Sınıflandırma | Medikal |
| Parkinson's | Binary Sınıflandırma | Medikal |
| Iris | Multi Sınıflandırma | Genel |
| Digits | Multi Sınıflandırma | Genel |
| Water Quality | Binary Sınıflandırma | Çevre |
| Air Quality | Regresyon | Çevre |
| Wastewater | Multi Sınıflandırma | Çevre |
| EEG Motor | Multi Sınıflandırma | Biyosinyal |
| EMG Hand | Multi Sınıflandırma | Biyosinyal |
| Chest X-Ray | Binary Sınıflandırma | Görüntü |
| Biodegradability | Binary Sınıflandırma | İlaç Keşfi |
| Liver Disease | Binary Sınıflandırma | Medikal |

### Tipik ML Workflow

1. Kullanıcı doğal dilde proje tarif eder
2. Agent veri setini bulur ve yükler
3. Proje yapısı oluşturulur: `data/`, `src/`, `results/`
4. En az **3 model** eğitilir ve karşılaştırılır
5. **5-fold cross validation** yapılır
6. Grafikler oluşturulur: confusion matrix, ROC curve, feature importance vb.
7. `report.md` ve `README.md` yazılır

### Oluşturulan Grafikler

- Confusion Matrix (normal + normalized)
- ROC Curve
- Feature Importance
- Korelasyon Matrisi (heatmap)
- Learning Curve
- Class Distribution

### Örnek Kullanım

```
>>> PROJECT: kanser Breast cancer veri setini kullanarak bir sınıflandırma modeli oluştur. 
    En az 5 model karşılaştır ve en iyi modeli seç.
```

---

## 10. 🧬 Biyomühendislik Araçları

### Protein Analizi

```python
from bioeng_toolkit import ProteinAnalyzer

pa = ProteinAnalyzer("MKWVTFISLLLLFSSAYS")
print(pa.summary())            # Kapsamlı özet
print(pa.molecular_weight())   # Moleküler ağırlık
print(pa.amino_acid_composition())  # Amino asit kompozisyonu
print(pa.hydropathy_profile())     # Hidrofobisite profili
print(pa.isoelectric_point())      # pI tahmini
print(pa.secondary_structure_tendency())  # İkincil yapı eğilimi
```

### Genomik Analiz

```python
from bioeng_toolkit import GenomicAnalyzer

ga = GenomicAnalyzer("ATGCGATCGATCG")
print(ga.gc_content())        # GC içeriği
print(ga.complement())        # Tamamlayıcı zincir
print(ga.reverse_complement())  # Ters tamamlayıcı
print(ga.transcribe())        # mRNA
print(ga.translate())         # Protein sekansı
print(ga.find_orfs())         # Açık okuma çerçeveleri
print(ga.melting_temperature()) # Erime sıcaklığı
```

### Atık Su Analizi

```python
from bioeng_toolkit import WastewaterAnalyzer

# Atık su kalite parametreleri analizi
ww = WastewaterAnalyzer()
# pH, BOD, COD, TSS gibi parametreleri analiz eder
```

### İlaç / Molekül Analizi

```python
from bioeng_toolkit import DrugMolecule

# SMILES tabanlı molekül analizi
mol = DrugMolecule("CCO")  # Etanol
print(mol.summary())
```

---

## 11. 🔌 Plugin Sistemi

### Plugin Oluşturma

`plugins/` klasörüne yeni bir `.py` dosyası ekleyin:

```python
# plugins/my_tool.py
from plugin_manager import ToolPlugin
from pathlib import Path

class MyCustomTool(ToolPlugin):
    @property
    def name(self):
        return "MYTOOL"

    @property
    def description(self):
        return "Benim özel aracım"

    def execute(self, payload: str, workspace: Path) -> str:
        # Tool mantığınız
        return f"Çıktı: {payload}"
```

Plugin otomatik olarak keşfedilecek ve agent'a `<MYTOOL>...</MYTOOL>` şeklinde kayıt edilecektir.

### Mevcut Plugin'ler

| Plugin | Dosya | Açıklama |
|--------|-------|----------|
| Örnek Plugin | `example_plugin.py` | LISTDIR + SYSINFO tool'ları |

---

## 12. 🧠 LLM Backend Değiştirme

### Desteklenen Backend'ler

| Backend | Ayar | Gereksinimler |
|---------|------|---------------|
| **Ollama** (varsayılan) | Yerel | Ollama sunucusu |
| **OpenAI** | API Key | `OPENAI_API_KEY` ortam değişkeni |
| **Anthropic** | API Key | `ANTHROPIC_API_KEY` ortam değişkeni |
| **Google Gemini** | API Key | `GOOGLE_API_KEY` ortam değişkeni |
| **HuggingFace** | API/Yerel | `HF_API_TOKEN` ortam değişkeni |

### Backend Değiştirme

`config.yaml`'da model değiştirin:

```yaml
agent:
  model: "gpt-4"  # OpenAI kullanmak için
```

Veya CLI'dan:

```bash
python3 agent.py --model gpt-4
```

API key'lerinizi Dashboard üzerinden de ayarlayabilirsiniz:
1. Dashboard'u başlatın (`python3 dashboard.py`)
2. **Ayarlar** sekmesine gidin
3. İlgili API key alanını doldurup kaydedin

---

## 13. 📜 Konuşma Geçmişi

### Otomatik Kayıt

Her kullanıcı mesajı, asistan yanıtı ve tool çalıştırmasından sonra oturum **otomatik olarak** kaydedilir.

### Geçmiş Dosyaları

Oturumlar `conversation_history/` klasöründe JSON olarak saklanır:

```
conversation_history/
├── 20260220_150000_a1b2c3d4.json
├── 20260221_033500_e5f6g7h8.json
└── ...
```

### Oturum Yönetimi

```bash
# Geçmiş listele
>>> /history

# Oturum yükle
>>> /load 20260220_150000_a1b2c3d4

# Yeni oturum başlat
>>> /new

# Mevcut oturumu kaydet
>>> /save

# Oturum sil
>>> /delete 20260220_150000_a1b2c3d4

# Oturum bilgileri
>>> /info
```

---

## 14. 🔍 Sorun Giderme

### Sık Karşılaşılan Sorunlar

#### ❌ Ollama bağlantı hatası

```
❌ LLM bağlantı hatası (model=qwen2.5:7b-instruct)
```

**Çözüm:**
```bash
# Ollama çalışıyor mu kontrol edin:
ollama serve

# Model yüklü mü:
ollama list

# Modeli indirin:
ollama pull qwen2.5:7b-instruct
```

#### ❌ `python` komutu bulunamıyor

```bash
# python3 kullanın veya alias oluşturun:
alias python=python3

# Veya venv aktifleştirin:
source venv/bin/activate
```

#### ❌ Modül bulunamıyor (ImportError / ModuleNotFoundError)

```bash
# Venv'in aktif olduğundan emin olun:
source venv/bin/activate

# Bağımlılıkları yeniden yükleyin:
pip install -r requirements.txt
```

> **Not:** Proje klasörü kopyalandıysa veya taşındıysa venv bozulur.
> Bu durumda venv'i yeniden oluşturun:
> ```bash
> rm -rf venv
> python3 -m venv venv
> source venv/bin/activate
> pip install -r requirements.txt
> ```

#### ❌ Timeout hatası

```bash
# Timeout süresini artırın:
python3 agent.py --timeout 300
```

#### ❌ Web araması engellendi

Mesajınıza `ALLOW_WEB_SEARCH` ekleyin veya `config.yaml`'da:

```yaml
security:
  allow_web_search: true
```

#### ❌ Dashboard başlatılamıyor

```bash
# Flask'ın yüklü olduğundan emin olun:
pip install flask

# Dashboard'u çalıştırın:
python3 dashboard.py
# http://localhost:5050 adresini ziyaret edin
```

### Log Dosyaları

Hatalar `logs/agent.log` dosyasında kayıtlıdır:

```bash
# Son logları görüntüle
>>> /logs 50

# veya doğrudan
tail -50 logs/agent.log
```

---

## 15. 📚 Komut Referansı

### Terminal Komutları

| Komut | Açıklama |
|-------|----------|
| `python3 agent.py` | CLI modunda başlat |
| `python3 web_ui.py` | Gradio web arayüzünü başlat |
| `python3 dashboard.py` | Flask görev panosunu başlat |
| `python3 -m pytest tests/` | Testleri çalıştır |

### Agent İç Komutları

| Komut | Açıklama |
|-------|----------|
| `/history` | Oturum listesi |
| `/load <id>` | Oturum yükle |
| `/new` | Yeni oturum |
| `/save` | Kaydet |
| `/delete <id>` | Oturum sil |
| `/info` | Oturum bilgisi |
| `/logs [N]` | Log görüntüle |
| `/help` | Yardım |
| `exit` / `quit` | Çıkış |

### Özel Anahtar Kelimeler

| Anahtar | Açıklama |
|---------|----------|
| `PROJECT: <ad>` | Proje adı belirle |
| `ALLOW_WEB_SEARCH` | Web aramayı etkinleştir |

### Dosya Yapısı Özeti

```
ai-agent/
├── agent.py                  # Ana agent kodu (CLI arayüzü + tool motoru)
├── bioeng_toolkit.py         # Biyomühendislik araç seti
├── config.yaml               # Merkezi yapılandırma dosyası
├── dashboard.py              # Flask görev panosu sunucusu
├── dataset_catalog.py        # Veri seti kataloğu (15+ hazır veri seti)
├── exceptions.py             # Özel hata sınıfları
├── llm_backend.py            # Çoklu LLM backend desteği
├── mlflow_tracker.py         # MLflow entegrasyonu
├── plugin_manager.py         # Plugin yükleme sistemi
├── progress.py               # Terminal spinner göstergesi
├── report_generator.py       # Otomatik ML rapor oluşturucu
├── requirements.txt          # Bağımlılık listesi
├── web_ui.py                 # Gradio web arayüzü
├── RAPOR.md                  # Proje durum raporu
├── KULLANMA_KILAVUZU.md      # Bu dosya
│
├── static/                   # Dashboard ön yüz dosyaları
│   └── dashboard.html        # Dashboard HTML/CSS/JS
│
├── utils/                    # Yardımcı modüller
│   ├── config.py             # Yapılandırma yönetimi
│   ├── model_compare.py      # Çoklu model karşılaştırma
│   └── visualize.py          # Görselleştirme araçları
│
├── plugins/                  # Eklenti (plugin) klasörü
│   └── example_plugin.py     # Örnek eklenti
│
├── tests/                    # Birim testler
│   ├── conftest.py           # Test yapılandırması
│   ├── test_agent.py         # Agent testleri
│   ├── test_exceptions.py    # Hata sınıf testleri
│   └── test_progress.py      # Spinner testleri
│
├── workspace/                # ML proje çalışma alanı
│   ├── breast_cancer_project/
│   ├── scratch_project/
│   └── wine_quality/
│
└── venv/                     # Python sanal ortamı
```

---

> *Bu kılavuz 22 Şubat 2026 tarihinde Bio-ML Agent v3.0 için oluşturulmuştur.*
