# Makine Öğrenimi Model Karşılaştırma Raporu

## 1. Giriş
Bu rapor, Pima Indian Diabetes veri seti üzerinde çeşitli makine öğrenimi modellerinin performansını değerlendirmektedir. Amaç, diyabet teşhisi için en uygun modeli belirlemek ve modelin kararlarını açıklayan içgörüler sunmaktır.

## 2. Veri Seti
Kullanılan veri seti, Pima Indian Diabetes veri setidir. Bu veri seti, Pima Kızılderililerindeki diyabet teşhisi ile ilgili sekiz klinik özelliği (örn. hamilelik sayısı, glikoz seviyesi, kan basıncı, BMI) ve bir çıktı değişkeni (diyabet varlığı/yokluğu) içermektedir.
Veri ön işleme adımı olarak, 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI' sütunlarındaki 0 değerleri NaN ile değiştirilmiş ve ardından medyan değerlerle doldurulmuştur.

## 3. Metodoloji
Veri seti %80 eğitim ve %20 test olmak üzere ikiye ayrılmıştır. Her model, `StandardScaler` ile birlikte bir `Pipeline` içinde eğitilmiştir. Modellerin performansını değerlendirmek için Accuracy, Precision, Recall, F1-Score ve ROC-AUC metrikleri kullanılmıştır. Ayrıca 5-katlı çapraz doğrulama uygulanmıştır.

## 4. Model Karşılaştırma Sonuçları
Aşağıdaki tabloda eğitilen modellerin test seti üzerindeki performans metrikleri ve çapraz doğrulama skorları gösterilmiştir:

| Model                      |   Accuracy |   Precision |   Recall |   F1-Score |   ROC-AUC |   CV_Mean_Score |   CV_Std_Score |
|:---------------------------|-----------:|------------:|---------:|-----------:|----------:|----------------:|---------------:|
| LogisticRegression         |   0.694805 |    0.574468 | 0.5      |   0.534653 |  0.812778 |        0.843068 |      0.0190711 |
| RandomForestClassifier     |   0.779221 |    0.717391 | 0.611111 |   0.66     |  0.81787  |        0.81887  |      0.0227842 |
| GradientBoostingClassifier |   0.75974  |    0.688889 | 0.574074 |   0.626263 |  0.83037  |        0.81867  |      0.018382  |
| SVC                        |   0.74026  |    0.652174 | 0.555556 |   0.6      |  0.796389 |        0.833951 |      0.0223606 |
| KNeighborsClassifier       |   0.753247 |    0.66     | 0.611111 |   0.634615 |  0.788611 |        0.786845 |      0.0389363 |

Yukarıdaki tabloya göre, en yüksek ROC-AUC değerine sahip model **GradientBoostingClassifier** olarak belirlenmiştir.

## 5. Görselleştirmeler
Seçilen en iyi model için aşağıdaki görselleştirmeler oluşturulmuştur:
- **Confusion Matrix (Normal & Normalize Edilmiş):** `results/plots/best_model_confusion_matrix.png`, `results/plots/best_model_confusion_matrix_normalized.png`
- **ROC Eğrisi:** `results/plots/best_model_roc_curve.png`
- **Özellik Önem Derecesi/Katsayıları:** `results/plots/best_model_feature_importance.png` veya `results/plots/best_model_feature_coefficients.png`
- **Veri Seti Korelasyon Matrisi:** `results/plots/correlation_matrix.png`
- **Test Seti Sınıf Dağılımı:** `results/plots/class_distribution_bar.png`, `results/plots/class_distribution_donut.png`

## 6. Açıklanabilir Yapay Zeka (XAI)
Modelin kararlarını anlamak için SHAP ve LIME yöntemleri kullanılmıştır:
- **SHAP Summary Plot:** `results/plots/shap_summary_plot.png` (Genel özellik önemini gösterir)
- **LIME Instance Açıklaması:** `results/plots/lime_explanation_instance_0.png` ve `results/plots/lime_explanation_instance_0.txt` (Belirli bir örnek için modelin neden o tahmini yaptığını gösterir)

### Klinik Karar Özeti

En iyi model olan **GradientBoostingClassifier** kullanılarak yapılan analizde, SHAP değerlerine göre diyabet riskini etkileyen en önemli özellikler şunlardır:

Bu özellikler, klinik uygulamalarda hastalığın erken teşhisi ve risk faktörlerinin belirlenmesi için yol gösterici olabilir.

## 7. Model Kullanımı
Eğitilmiş en iyi model `results/best_model.pkl` dosyasında kaydedilmiştir. Model aşağıdaki gibi yüklenebilir ve yeni veriler üzerinde tahminler yapmak için kullanılabilir:

```python
import joblib
import pandas as pd

best_model = joblib.load('results/best_model.pkl')

# Yeni veri örneği (sütun isimleri eğitimdekiyle aynı olmalı)
new_data = pd.DataFrame([[6, 148, 72, 35, 0, 33.6, 0.627, 50]], columns=[
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
    'BMI', 'DiabetesPedigreeFunction', 'Age'
])

prediction = best_model.predict(new_data)
prediction_proba = best_model.predict_proba(new_data)[:, 1]

print(f'Tahmin: {prediction[0]} (0: Diyabet Yok, 1: Diyabet Var)')
print(f'Diyabet Var olma Olasılığı: {prediction_proba[0]:.4f}')
```
