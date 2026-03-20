# 📖 Bio-ML Agent — Kullanma Kılavuzu

> **Sürüm:** v0.1.0-clean (Production Ready)
> **Tarih:** 20 Mart 2026  
> **Python:** 3.11+  
> **İşletim Sistemi:** Linux / macOS

---

## 📋 İçindekiler

1. [Kurulum](#1--kurulum)
2. [Hızlı Başlangıç](#2--hızlı-başlangıç)
3. [Terminal Arayüzü (CLI)](#3--terminal-arayüzü-cli)
4. [Web Arayüzü (Gradio)](#4--web-arayüzü-gradio)
5. [Görev Panosu (Web UI)](#5--görev-panosu-web-ui)
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
16. [V5 İleri Düzey Özellikleri (WhatsApp, Ses, Görüntü & RAG)](#16--v5-ileri-düzey-özellikler-whatsapp-ses-görüntü--rag)
17. [Swarm Çoklu Ajan Mimarisi](#17--swarm-çoklu-ajan-mimarisi)
18. [Açıklanabilir Yapay Zeka (XAI)](#18--açıklanabilir-yapay-zeka-xai)
19. [Sürekli Öğrenme ve Veri Akışları](#19--sürekli-öğrenme-ve-veri-akışları)
21. [Güvenlik ve HITL (Human-in-the-Loop)](#21--güvenlik-ve-hitl-human-in-the-loop)
22. [Kalıcı İş Akışları (Temporal)](#22--kalıcı-iş-akışları-temporal)

---

## 1. 🔧 Kurulum

### Ön Gereksinimler

- Python 3.11 veya üstü
- [Ollama](https://ollama.ai/) (yerel LLM için)
- pip (Python paket yöneticisi)

### Adım Adım Kurulum

```bash
# 1. Proje dizinine gidin
cd bio-ml-agent_v0.2/

# 2. Sanal ortamı oluşturun (zaten varsa bu adımı atlayın)
python3 -m venv venv

# 3. Sanal ortamı aktifleştirin
source venv/bin/activate

# 4. Bağımlılıkları yükleyin (Tüm özellikler için)
pip install -e ".[all]"

# 5. Ollama modelini indirin (Ollama çalışır durumda olmalı)
ollama pull qwen2.5:7b-instruct
```

> **⚠️ Önemli Not:** Proje klasörü kopyalandıysa veya taşındıysa venv bozulabilir.
> Bu durumda eski venv'i silip yeniden oluşturun:
> ```bash
> rm -rf venv
> python3 -m venv venv
> source venv/bin/activate
> pip install -e ".[all]"
> ```

### Bağımlılıklar

| Paket | Amaç |
|-------|------|
| `litellm` | Multi-LLM Proxy ve Router |
| `qdrant-client` | Agentic Vektör Hafızası |
| `temporalio` | Durable Execution / Workflow |
| `langgraph` | Ajan Orkestrasyonu (State Control) |
| `playwright` | Otonom Browser Driver |
| `cryptography` | Vault (Kasa) Şifreleme |
| `opentelemetry` | Gözlemlenebilirlik (Traces/Metrics) |
| `scikit-learn` | ML modelleri |
| `pandas` | Veri işleme |
| `numpy==1.26.4` | Sayısal hesaplama (Pinli sürüm) |
| `matplotlib` | Grafikler |
| `pytest` | Testler |
| `gradio` | Web arayüzü |

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
🧠 Bio-ML Agent ready | model=mock-gpt | workspace=./workspace
📜 Oturum ID: 20260320_190000_a1b2c3d4
♻️ Oturum otomatik kurtarıldı: 20260320_185500_... (Eğer varsa)
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

### Otonom Hesap ve E-Posta Yönetimi (Yeni!)
Eğer sizden bir veri indirmesi veya kilitli bir alana girmesi istenirse, ajan `Mail.tm` altyapısını kullanarak **kendi kendine** geçici bir mail hesabı oluşturur, siteye (`BROWSER_AGENT` aracılığıyla) kayıt olur, gelen onay e-postasını okur ve şifresini/kullanıcı adını kendi dahili Vault (`~/.bio-ml-agent/vault.json`) kasasına kaydeder. Sonraki projelerinizde bu hazır bilgiyi kullanarak tekrar uğraşmadan API anahtarlarına vb. erişim sağlayabilir.

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
- **YENİ:** Mikrofon simgesi ile sesli komut verme (Voice Interface)
- **YENİ:** İmaj ve tıbbi belge yükleyerek görsel analiz (Vision API)
- **YENİ:** Data Explorer sekmesi ile Workspace'deki CSV ve interaktif Plotly HTML grafiklerini anında görüntüleme.
- Model, timeout, max_steps ayarlarını arayüzden değiştirme
- Yeni oturum başlatma
- Oturum listesini görüntüleme

---

## 5. 📊 Görev Panosu (Web UI)

Otonom ajanların görevleri ve projelerin durumu doğrudan **Web UI (Gradio)** içindeki özel sekmelerde (tabs) yönetilir. Eskiden harici bir uygulama olan dashboard, artık Web UI'ye tam entegredir.

### Web UI'yi Başlatma

```bash
source venv/bin/activate
python3 web_ui.py
```

Tarayıcınızda `http://localhost:7860` adresini açıp üst kısımdaki **Görevler (Dashboard)** sekmesine tıklayın.

### Görev Panosu Özellikleri

| Özellik | Açıklama |
|---------|----------|
| **Görev Yönetimi** | Ajan görevlerini tablo halinde görüntüleme |
| **Durum Filtreleme** | Bekleyen (PENDING), Çalışan (IN_PROGRESS), Biten (COMPLETED) görevleri ayıklama |
| **Seçme ve Aksiyon** | Görevi onaylama (Approve) veya Reddetme (Reject/Cancel) |
| **Proje İstatistikleri** | İlgili projeye ait özet bilgileri görüntüleme |
| **API Anahtar Yönetimi** | Ortam değişkenlerini UI üzerinden ayarlama |


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
  allow_web_search: true
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
| Web Open | `<WEB_OPEN>...</WEB_OPEN>` | URL'den statik metin çek |
| Browser Open | `<BROWSER_OPEN>...</BROWSER_OPEN>` | JavaScript render eden Chromium ile URL içeriği çek |
| Browser Action | `<BROWSER_ACTION>...</BROWSER_ACTION>` | Manuel Playwright komutları (tıklama, yazma vb.) |
| Browser Agent | `<BROWSER_AGENT>...</BROWSER_AGENT>` | **YENİ:** LLM destekli kendi kendine gezinen otonom tarayıcı asistanı |
| Read File | `<READ_FILE>...</READ_FILE>` | Dosya oku |
| Write File | `<WRITE_FILE>...</WRITE_FILE>` | Dosya yaz |
| TODO | `<TODO>...</TODO>` | Yapılacaklar listesi |

### 🤖 Otonom Browser Sub-Agent (YENİ)
Sürüm 7.0 ile birlikte agent, internet üzerindeki karmaşık araştırma ve laboratuvar veri toplama görevleri için kendi içinde bir **Browser Sub-Agent** çağırabilir. 
- Ajan Chromium altyapısıyla çalışır. Sayfayı açar, HTML öğelerine tıklar, arama kutularına yazar.
- Sizin `<BROWSER_AGENT>...</BROWSER_AGENT>` etiketini manuel kullanmanıza gerek yoktur. Ajan, "*Google'da p53 gen mutasyonlarını araştır*" dediğinizde bu aracı kendi inisiyatifiyle çalıştırıp sonuçları size raporlar.

![Browser Agent In Action](docs/feature_browser.png)

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

**Not:** v3.5 güncellemesi itibarıyla tüm bu grafikler artık statik PNG formatından ziyade, Data Explorer web arayüzünde fareyle üzerine gelip etkileşebildiğiniz (zoom in/out) interaktif **Plotly HTML** formatlarında kaydedilmektedir.

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

API key'lerinizi yapılandırma araçları (örn. `utils/config.py`) aracılığıyla ayarlayabilirsiniz. Artık Web UI'deki **Ayarlar** sekmesi ile de dinamik değiştirilebilir.

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

# RAG Bellek Arama Komutları
>>> /rag [kelime] # Geçmiş bellekte sorgu yaparak ilgili konuşmaları getirir
>>> /ragindex # Workspace dosyalarını vektör veritabanında kronolojik sırayla manuel indeksler
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
pip install -e ".[all]"
```

> **Not:** Proje klasörü kopyalandıysa veya taşındıysa venv bozulur.
> Bu durumda venv'i yeniden oluşturun:
> ```bash
> rm -rf venv
> python3 -m venv venv
> source venv/bin/activate
> pip install -e ".[all]"
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

#### ❌ Web UI (Gradio) başlatılamıyor

```bash
# Gradio yüklü mü emin olun:
pip install gradio

# Web UI çalıştırın:
python3 web_ui.py
# http://localhost:7860 adresini ziyaret edin
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
| `python3 agent.py` | CLI modunda başlat |
| `python3 web_ui.py` | Gradio web arayüzünü (Dashboard dahil) başlat |
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
├── dataset_catalog.py        # Veri seti kataloğu (15+ hazır veri seti)
├── exceptions.py             # Özel hata sınıfları
├── llm_backend.py            # Çoklu LLM backend desteği
├── mlflow_tracker.py         # MLflow entegrasyonu
├── plugin_manager.py         # Plugin yükleme sistemi
├── progress.py               # Terminal spinner göstergesi
├── report_generator.py       # Otomatik ML rapor oluşturucu
├── pyproject.toml            # Modern proje ve bağımlılık yönetimi
├── web_ui.py                 # Gradio web arayüzü
├── RAPOR.md                  # Proje durum raporu
├── KULLANMA_KILAVUZU.md      # Bu dosya
│
├── static/                   # Ön yüz dosyaları
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

## 16. 🚀 V5 İleri Düzey Özellikler (WhatsApp, Ses, Görüntü & RAG)

Bio-ML Agent artık V5 yol haritasıyla birlikte çok boyutlu (multimodal) çalışma kabiliyetlerine sahip olmuştur.

### 📱 WhatsApp Üzerinden Uzaktan Kontrol

Agent, Node.js üzerinden `whatsapp-web.js` köprüsü kurarak direkt telefonunuzdan ML projeleri kurmanızı ve eğitmenizi sağlar.
*Bu özellik telefonunuzun asistan üzerinden projeler denerken laboratuvara bağlı kalmanızı engeller.*

**Çalıştırmak İçin:**
```bash
./start_whatsapp_bot.sh
```
1. Çıkan QR kodu WhatsApp cihaz bağlama özelliğiyle okutun (Cihaza bağlıyken agent arka planda çalışmaya hazır bekler).
2. WhatsApp sohbetinizden şunu yazın: `"Gemini, bana diyabet veri setini alıp Random Forest modelini eğiten bir proje kur. En iyi sonuçları buraya rapor et."`
3. Agent işlemi bitirdiğinde sonucun özetini Markdown sentezi olarak telefonunuza gönderecektir.

### 🎙️ Sesli Komut & 👁️ Görüntü İşleme (Gradio Web UI)

`web_ui.py`'yi başlattığınızda arayüzde artık aşağıdaki yetenekler eklidir:
- **Mikrofon Modülü:** Uzun promptlar yazmak yerine mikrofon tuşuna basıp sesli olarak komut verebilirsiniz.
- **Multimodal (Vision) Dosya Yükleme:** Arayüzün sol altındaki ataç simgesinden (veya sürükle bırak ile) MRI görüntüleri, hücre boyamaları veya analiz makalesi (PDF/Resim) ekleyebilirsiniz. Ajan Gemini'ın multimodality uç noktasını (Vision) kullanarak görseli okuyup direkt hastalık teşhisi yapabilir veya çıkarım elde edebilir.

### 🧠 Otomatik Kalıcı Uzun Dönem Hafıza (Qdrant RAG)

- Ajan her turn (tur) tamamladığında konuşmanızı arka planda **Qdrant** vektör veritabanına indeksler.
- **Provenance (Kaynak Takibi):** Bilginin hangi dosyadan veya web sayfasından geldiği her zaman kaydedilir.
- **TTL (Yaşam Süresi):** Hassas veriler belirli bir süre sonra bellekten otomatik temizlenecek şekilde ayarlanabilir.
- **Hibrit Arama:** Hem anahtar kelime hem de anlamsal (semantic) arama birleştirilerek en doğru bağlam getirilir.

---

## 17. 🧩 Swarm Çoklu Ajan Mimarisi

V6+ itibarıyla Bio-ML Agent, tek bir monolitik LLM yerine **3 uzman alt-ajan** koordinasyonu ile çalışır.

### Ajan Rolleri (LangGraph Orchestration)

| Ajan | Görev |
|------|-------|
| **LangGraph Orchestrator** | Planla -> Uygula -> Doğrula döngüsünü yöneten ana zeka. |
| **Data Engineer** | Veri yükleme, temizleme ve hazırlama uzmanı. |
| **ML Expert** | Model eğitimi, hiperparametre ve XAI uzmanı. |
| **Bioinfo Expert** | Klinik yorumlama ve biyolojik anlamlandırma uzmanı. |

### Kullanım

```python
from swarm.orchestrator import SwarmOrchestrator
from utils.config import load_config

cfg = load_config()
orchestrator = SwarmOrchestrator(cfg)
response = orchestrator.process([{"role": "user", "content": "Diyabet verisini analiz et"}])
```

### Demo

```bash
source venv/bin/activate
export GEMINI_API_KEY="YOUR_KEY"
python scripts/demos/swarm_diabetes_demo.py
```

---

## 18. 🔍 Açıklanabilir Yapay Zeka (XAI)

ML Uzmanı, model eğitimini bitirdikten sonra otomatik olarak `xai_engine.py` üzerinden SHAP ve LIME analizi çalıştırır.

### Üretilen XAI Çıktıları

| Grafik | Açıklama |
|--------|----------|
| SHAP Summary Plot (Bar) | Özellik önem sıralaması |
| SHAP Summary Plot (Beeswarm) | Her özelliğin bireysel etkileri |
| SHAP Force Plot | Tek hasta için karar açıklaması |
| SHAP Dependence Plot | Özellik-tahmin ilişkisi |

### Gradio XAI Sekmesi

Web arayüzünde **🔍 Açıklanabilirlik (XAI)** sekmesine geçerek:
1. "XAI Grafikleri Yenile" butonuna tıklayın
2. Ajan tarafından üretilen tüm SHAP/LIME grafiklerini galeri olarak görüntüleyin

---

## 19. 🔄 Sürekli Öğrenme ve Veri Akışları

Sistem artık tek seferlik CSV yüklemelerinin ötesinde, **canlı veri kaynaklarından** beslenir.

### Veritabanı Bağlantısı

```python
from data_streams.db_connector import DBConnector

db = DBConnector("postgresql://user:pass@localhost/health_db")
new_data = db.get_new_data_since("patients", "created_at", "2026-03-01")
```

### Redis Streams (Gerçek Zamanlı Sensör Verisi)

```python
from data_streams.kafka_redis_consumer import StreamConsumer

consumer = StreamConsumer("sensor_stream", "ml_group", "worker_1")
consumer.listen(batch_size=10, callback=retrain_model)
```

### Active Learning Worker

```bash
# Arkaplanda dinleyici başlat (yeni veri gelince otomatik retrain)
python swarm/active_learning_worker.py
```

### Demo

```bash
python scripts/demos/active_learning_demo.py
```

---

## 20. 🐳 Docker Compose ile Dağıtım

Sistem 7 mikroservisten oluşur ve sıkılaştırılmış güvenlik politikalarıyla çalışır:

| Servis | Port | Açıklama |
|--------|------|----------|
| `redis` | 6380 | Mesaj kuyruğu ve cache (Temporal/RQ) |
| `api` | 8001 | FastAPI REST sunucusu + Webhook |
| `worker` | — | Arkaplan işçisi (Temporal / RQ / LangGraph) |
| `web_ui` | 7860 | Gradio web arayüzü |
| `mlflow` | 5005 | MLflow Tracking Server |
| `litellm` | 4000 | Multi-LLM Proxy & Routing Gateway |
| `qdrant` | 6333 | Agentic Vektör Hafızası (RAG) |

### Güvenlik ve İzolasyon (S9-1)
- **Non-Root User:** Konteynerler `agent` kullanıcısı ile kısıtlı yetkilerle çalışır.
- **Seccomp:** Gereksiz sistem çağrıları engellenmiştir.
- **No-New-Privileges:** Privilege escalation saldırıları önlenmiştir.

### Çalıştırma

```bash
# Tüm servisleri başlat
docker-compose up -d

# Logları izle
docker-compose logs -f worker
```

---

## 21. 🛡️ Güvenlik ve HITL (Human-in-the-Loop)

Bio-ML Agent v1.0, otonom yeteneklerini güvenlik bariyerleriyle dengeler.

### 🛡️ Sandbox Runtime
Ajanın çalıştırdığı tüm Python kodları, CPU ve bellek limitli izole bir `SandboxRuntime` üzerinde yürütülür. Bu, sistem kaynaklarının aşırı tüketimini ve yetkisiz erişimleri engeller.

### 🤝 Human-in-the-Loop (HITL)
Kritik operasyonlar öncesinde ajan durur ve sizden onay ister:
- **Tehlikeli Bash Komutları:** Dosya silme (`rm`), sistem ayarları.
- **Harici API Kayıtları:** Web üzerinden yeni hesap açma veya API key edinme.
- **Yüksek Maliyetli İşlemler:** Çok yüksek token tüketimi gerektiren analizler.

### 🔒 Audit Trail (Denetim İzleri)
Tüm kritik eylemler `audit_logs/` klasöründe zaman damgalı JSONL formatında saklanır. Bu kayıtlar değiştirilemez ve sistemin neyi neden yaptığını denetlemenizi sağlar.

---

## 22. ⏳ Kalıcı İş Akışları (Temporal)

Bio-ML Agent, uzun süren analizleri (örn. büyük bir genomik veri setinin taranması) yönetmek için **Temporal** kullanır.

### Avantajları:
- **Dayanıklılık (Durability):** Bilgisayarınız kapansa veya internet kesilse bile, Temporal workflow'u kaldığı yerden devam ettirir.
- **Takip Edilebilirlik:** Workflow geçmişini (event history) inceleyerek ajanın her adımını görebilirsiniz.
- **Kill-Switch:** İstediğiniz an çalışan aktif bir analizi durdurabilir, iptal edebilir veya baştan başlatabilirsiniz.

---

> *Bu kılavuz 4 Mart 2026 tarihinde Bio-ML Agent v1.0 (Ultra-Ajan Upgrade) için derlenmiştir.*
