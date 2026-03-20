# Golden Path Demo Senaryosu 🏆

Bu senaryo, Bio-ML Agent'ın yeteneklerini uçtan uca göstermek için tasarlanmıştır.

## 📋 Senaryo Adımları

Aşağıdaki promptları sırayla ajana göndererek demoyu gerçekleştirebilirsiniz:

### 1. Veri Hazırlama ve Keşif
**Prompt:**
> `examples/golden_path_demo/breast_cancer_data.csv` dosyasını oku, temel istatistikleri çıkar ve 'diagnosis' hedef değişkenine göre korelasyon analizi yap. Sonuçları özetle.

### 2. Model Eğitimi ve Optimizasyon
**Prompt:**
> Bu veriyi kullanarak 'diagnosis' değişkenini tahmin eden bir RandomForest modeli eğit. Veriyi %80 eğitim, %20 test olarak böl. Accuracy ve F1 skorlarını raporla.

### 3. Açıklanabilirlik (XAI)
**Prompt:**
> Eğittiğin modelin tahminlerini etkileyen en önemli 5 özelliği (feature importance) belirle. Bu özelliklerin biyolojik/klinik açıdan ne ifade edebileceğini yorumla.

### 4. Raporlama
**Prompt:**
> Tüm bu analiz adımlarını, metrikleri ve model yorumlarını içeren profesyonel bir `demo_report.md` dosyası oluştur.

---
> [!TIP]
> Bu demoyu `MockBackend` ile denemek isterseniz, `web_ui.py` başlatırken model ismine `mock-demo` diyebilirsiniz. Gerçek analizler için OpenAI, Anthropic veya Gemini anahtarlarınızı kullanın.
