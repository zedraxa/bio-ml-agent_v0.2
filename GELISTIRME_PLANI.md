# 🗺️ Bio-ML Agent — İyileştirme & Geliştirme Yol Haritası

> **Tarih:** 28 Şubat 2026  
> **Mevcut Durum:** v3 — 159 test, 13 modül, Gemini entegrasyonu tamamlandı

---

## 🔴 Yüksek Öncelik

### 1. Model Kaydetme & Yükleme (joblib) (Tamamlandı)
- `utils/model_compare.py`'ye `save_best_model()`, `save_all_models()`, `load_model()` eklendi
- `utils/model_loader.py` standalone model yükleme utility'si oluşturuldu
- `agent.py` SYSTEM_PROMPT'una model kaydetme/yükleme talimatları eklendi
- `tests/test_model_save_load.py` ile doğrulandı

### 2. Eksik Modül Testleri (Tamamlandı)
- **Hedef:** Test sayısını 159 → 250+ çıkarmak ✅ (329 test)

| Modül | Test Var mı? | Yazılacak Testler |
|---|---|---|
| `llm_backend.py` | ✅ | Mock LLM ile chat(), backend seçimi, hata yönetimi |
| `dataset_catalog.py` | ✅ | 15 veri setinin doğru yüklendiği, hatalı isim kontrolü |
| `utils/model_compare.py` | ✅ | compare_models() doğruluğu, edge case'ler |
| `utils/visualize.py` | ✅ | Grafik dosyalarının oluşturulup oluşturulmadığı |
| `web_ui.py` | ✅ | Gradio bileşenlerinin başlatılması |
| `report_generator.py` | ✅ | Rapor çıktı formatı doğrulama |
| `plugin_manager.py` | ✅ | Plugin keşfi, yükleme, çalıştırma |

### 3. Entegrasyon (E2E) Testleri (Tamamlandı)
- **Sorun:** Unit testler parça parça çalışıyor ama agent'ın komple proje üretip üretemediği test edilmiyor
- **Çözüm:** Mock LLM ile tam döngü testi: prompt → tool çalıştır → dosya oluştur → doğrula
- **Dosya:** `tests/test_e2e.py`

---

## 🟡 Orta Öncelik

### 4. CI/CD Pipeline (GitHub Actions) (Tamamlandı)
- Her push'ta otomatik test çalıştırma
- Dosya: `.github/workflows/test.yml`
- İçerik: Python kurulumu → pip install → pytest çalıştır → sonuç raporla

### 5. Hiperparametre Optimizasyonu (Tamamlandı)
- `GridSearchCV` ve `RandomizedSearchCV` entegrasyonu yapıldı
- `utils/hyperparameter_optimizer.py` modülü oluşturuldu
- `agent.py` SYSTEM_PROMPT'una hiperparametre optimizasyonu talimatları eklendi

### 6. Veri Ön İşleme Pipeline'ı
- Eksik değer (NaN) otomatik doldurma stratejileri
- Outlier tespiti ve temizleme
- Feature engineering: polinom özellikler, PCA boyut indirgeme
- **Dosya:** Yeni `utils/preprocessor.py`

### 7. Docker Desteği (Tamamlandı)
- `Dockerfile` + `docker-compose.yml` oluştur
- Ollama ve agent'ı tek komutla ayağa kaldır
- Efor: ~2 saat

### 8. Dashboard İyileştirmeleri
- `dashboard.py` ile agent entegrasyonu
- Canlı grafik izleme, model karşılaştırma paneli
- Proje geçmişi ve sonuç görüntüleme
- **Dosyalar:** `dashboard.py`, `static/dashboard.html`

---

## 🟢 Düşük Öncelik (İleri Seviye)

### 9. RAG (Retrieval-Augmented Generation) (Tamamlandı)
- Agent'ın önceki projeleri ve raporları arayarak yanıt vermesi
- Vektör veritabanı (ChromaDB/FAISS) entegrasyonu
- **Dosyalar:** Yeni `rag_engine.py`

### 10. Multi-Agent Kolaborasyonu (Tamamlandı)
- Veri analizi, model seçimi ve rapor yazımı için uzmanlaşmış alt-agent'lar eklendi (`multi_agent.py`)
- Orchestrator agent koordinasyonu `agent.py`'e tanımlandı

### 11. REST API Modu (Tamamlandı)
- `--mode api --port 8080` ile web servisi olarak çalıştırma
- POST `/api/chat` endpoint'i
- WebSocket ile gerçek zamanlı ilerleme bildirimi
- **Dosya:** Yeni `api_server.py` hazırlikları tamamlandı.

### 12. Biyomühendislik Toolkit Entegrasyonu (Tamamlandı)
- `bioeng_toolkit.py`'deki analiz araçları agent'ın `<PYTHON>` kullanım yeteneğine entegre edildi.
- Protein, genomik, atık su ve medikal görüntü analizleri için testler eklendi ve sistem promptu güncellendi.
- **Dosyalar:** `agent.py`, `tests/test_bioeng_toolkit_integration.py`

---

## 🧪 Yürütülmesi Gereken Test Senaryoları

### Unit Testler (Tamamlandı — Tüm Modüller)
```
tests/test_llm_backend.py
  - test_gemini_backend_init()           → API key yokken hata fırlatır mı
  - test_ollama_backend_chat_mock()      → Mock yanıtla chat çalışır mı
  - test_auto_backend_selection()        → Model adına göre doğru backend seçilir mi
  - test_connection_error_handling()     → API hatalarında LLMConnectionError fırlatılır mı

tests/test_dataset_catalog.py
  - test_load_breast_cancer()            → breast_cancer verisi yüklenir mi
  - test_load_all_datasets()             → Tüm 15+ veri seti yüklenir mi
  - test_invalid_dataset_name()          → Geçersiz isimde hata fırlatır mı
  - test_dataset_shape()                 → Dönen X, y boyutları doğru mu

tests/test_model_compare.py
  - test_compare_classification()        → 5 model karşılaştırması çalışır mı
  - test_compare_regression()            → Regresyon görevi çalışır mı
  - test_output_json()                   → JSON çıktı formatı doğru mu
  - test_best_model_selection()          → En iyi model doğru seçilir mi

tests/test_visualize.py
  - test_confusion_matrix_png()          → PNG dosyası oluşturulur mu
  - test_roc_curve_png()                 → ROC curve oluşturulur mu
  - test_all_plots()                     → 6 grafik birden oluşturulur mu
  - test_output_directory_creation()     → Klasör yoksa otomatik oluşturulur mu

tests/test_path_strip.py
  - test_workspace_prefix_strip()        → workspace/ silinir mi
  - test_double_nesting_strip()          → workspace/proj/workspace/proj/ düzeltilir mi
  - test_known_roots_detection()         → src/, data/, results/ tanınır mı
  - test_known_files_detection()         → report.md, README.md tanınır mı
  - test_no_change_needed()             → Zaten doğru yol değişmez mi
```

### Entegrasyon Testleri
```
tests/test_e2e.py
  - test_full_project_creation_mock()    → Mock LLM ile tam proje oluşturma
  - test_write_file_path_integrity()     → Dosyalar doğru yere yazılır mı
  - test_bash_cwd_correctness()          → BASH komutları doğru CWD'den çalışır mı
  - test_conversation_save_load()        → Oturum kaydedilir ve yüklenebilir mi
```

### Güvenlik Testleri (Mevcut ama genişletilebilir)
```
  - test_path_traversal_block()          → ../../../etc/passwd engellenir mi
  - test_dangerous_command_block()       → rm -rf / engellenir mi
  - test_timeout_enforcement()           → Sonsuz döngü timeout ile kesilir mi
  - test_api_key_not_logged()            → API key'ler log dosyasına yazılmaz mı
```

---

## 📊 Hedef Metrikler

| Metrik | Şu An | Hedef |
|--------|-------|-------|
| Unit test sayısı | 159 | 250+ |
| Test coverage | ~60% | 85%+ |
| Modül testi olan dosya | 3/13 | 10/13 |
| CI/CD | ❌ | ✅ GitHub Actions |
| Docker | ❌ | ✅ Dockerfile |
| E2E test | ❌ | ✅ Mock LLM ile |

---

## 🎯 Önerilen Aksiyon Sırası

1. [x] `tests/test_llm_backend.py` yaz (mock testler)
2. [x] `tests/test_dataset_catalog.py` yaz
3. [x] `tests/test_model_compare.py` yaz
4. [x] `tests/test_path_strip.py` yaz
5. [x] `.github/workflows/test.yml` ekle (CI/CD)
6. [x] Model kaydetme (joblib) desteği ekle
7. [x] `Dockerfile` oluştur
8. [x] Dashboard entegrasyonu
9. [x] Hiperparametre optimizasyonu
10. [x] REST API modu

---

> *Bu dosya, projenin gelecek sürümlerinde referans noktası olarak kullanılabilir.*
