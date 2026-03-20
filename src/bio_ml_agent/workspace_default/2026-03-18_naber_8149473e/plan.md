1.  **Veri Kümesi Seçimi**: Biyomedikal bir sınıflandırma veri kümesi bulacağım (örn. kalp hastalığı).
2.  **Veri Kümesi İndirme**: Seçilen veri kümesini `data/raw/` klasörüne indireceğim.
3.  **Proje Yapısı Oluşturma**: `src/`, `data/raw/`, `results/`, `results/plots/` klasörlerini ve `requirements.txt` dosyasını oluşturacağım.
4.  **Yardımcı Araçları (Utils) Oluşturma**: `utils/model_compare.py`, `utils/preprocessor.py`, `utils/visualize.py`, `utils/hyperparameter_optimizer.py`, `utils/model_loader.py` ve `xai_engine.py` dosyalarını oluşturacağım.
5.  **Veri Ön İşleme ve Analiz**: Veriyi yükleyecek, eksik değerleri işleyecek, kategorik değişkenleri kodlayacak ve sayısal özellikleri ölçeklendireceğim. `utils/preprocessor.py` kullanacağım.
6.  **Model Eğitimi ve Karşılaştırma**: Logistic Regression, RandomForest ve Gradient Boosting gibi modelleri `utils/model_compare.py` ile eğitecek ve performanslarını karşılaştıracağım.
7.  **Hiperparametre Optimizasyonu**: En iyi performansı gösteren model için `utils/hyperparameter_optimizer.py` ile hiperparametre optimizasyonu yapacağım.
8.  **Görselleştirmeler Oluşturma**: `utils/visualize.py` kullanarak çeşitli grafikler (Confusion Matrix, ROC Curve, Feature Importance vb.) oluşturacağım.
9.  **Açıklayıcı Yapay Zeka (XAI)**: `xai_engine.py` kullanarak SHAP özet ve LIME açıklama grafikleri oluşturacağım.
10. **Raporlama**: `results/comparison_report.md`, `README.md` ve SHAP sonuçlarına dayalı "Klinik Karar Özeti"ni hazırlayacağım.