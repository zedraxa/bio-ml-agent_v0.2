# Kanser Hücresi Analizi Sonuç Raporu
Bu rapor, Wisconsin Meme Kanseri (Tanısal) veri setini kullanarak meme kanseri teşhisi için makine öğrenimi modellerinin performansını detaylandırmaktadır.

## 1. Veri Seti
Veri seti, çekirdek özelliklerinden elde edilen 30 adet sayısal özellik ile hücre çekirdeğinin dijitalleştirilmiş bir görüntüsünden elde edilmiştir. Hedef değişken, teşhis ('M' için Malign, 'B' için Benign) şeklindedir.

## 2. Ön İşleme Adımları
- Eksik değerler medyan ile doldurulmuştur.
- IQR yöntemi kullanılarak aykırı değerler tespit edilmiş ve eğitim veri setinden çıkarılmıştır.
- Özellikler StandardScaler kullanılarak standartlaştırılmıştır.

## 3. Model Karşılaştırması
Çeşitli sınıflandırma modelleri eğitilmiş ve 5 katlı çapraz doğrulama ile değerlendirilmiştir. Performans, Accuracy, Precision, Recall, F1 Score ve ROC AUC metrikleri üzerinden ölçülmüştür.

### Karşılaştırma Tablosu
| Model              |   Accuracy |   Precision |   Recall |   F1 Score |   ROC AUC |   CV_Accuracy_Mean |   CV_Accuracy_Std |
|:-------------------|-----------:|------------:|---------:|-----------:|----------:|-------------------:|------------------:|
| LogisticRegression |   0.973684 |    0.973711 | 0.973684 |   0.973616 |  0.993056 |           0.96875  |         0.0171163 |
| RandomForest       |   0.973684 |    0.974737 | 0.973684 |   0.973465 |  0.994048 |           0.946875 |         0.0233854 |
| GradientBoosting   |   0.95614  |    0.956869 | 0.95614  |   0.955776 |  0.988095 |           0.95625  |         0.030298  |
| SVM                |   0.938596 |    0.944037 | 0.938596 |   0.937229 |  0.992725 |           0.9625   |         0         |
| KNN                |   0.938596 |    0.944037 | 0.938596 |   0.937229 |  0.993717 |           0.9625   |         0.0390312 |

Karşılaştırma sonucunda **RandomForest** en iyi performansı gösteren model olarak belirlenmiştir.

## 4. Hiperparametre Optimizasyonu (RandomForest)
En iyi model olan **RandomForest** için Random Search ile hiperparametre optimizasyonu yapılmıştır.
### Optimize Edilmiş Parametreler:
- `model__n_estimators`: `50`
- `model__min_samples_split`: `2`
- `model__max_depth`: `10`

## 5. Görselleştirmeler
Aşağıdaki grafikler, optimize edilmiş en iyi modelin performansını ve veri özelliklerini daha ayrıntılı incelemektedir:

- **Confusion Matrix (Normalleştirilmiş ve Normal)**: Modelin doğru ve yanlış sınıflandırmalarını gösterir. ([scratch_project/results/plots/confusion_matrix.png], [scratch_project/results/plots/normalized_confusion_matrix.png] tesislerinde bulunabilir)
- **ROC Eğrisi**: Modelin farklı karar eşiklerinde True Positive Rate (TPR) ve False Positive Rate (FPR) arasındaki dengeyi gösterir. ([scratch_project/results/plots/roc_curve.png] tesisinde bulunabilir)
- **Özellik Önem Derecesi**: Modelin karar verirken hangi özelliklere daha çok güvendiğini gösterir. ([scratch_project/results/plots/feature_importance.png] tesisinde bulunabilir)
- **Korelasyon Matrisi**: Özellikler arasındaki ilişkiyi gösteren bir ısı haritası. ([scratch_project/results/plots/correlation_matrix.png] tesisinde bulunabilir)
- **Öğrenme Eğrisi**: Modelin eğitim boyutuyla performansının nasıl değiştiğini gösterir, aşırı uyum ve eksik uyum sorunlarını anlamaya yardımcı olur. ([scratch_project/results/plots/learning_curve.png] tesisinde bulunabilir)
- **Sınıf Dağılımı**: Hedef değişkenin sınıflarının veri setindeki dağılımını gösterir. ([scratch_project/results/plots/class_distribution_bar.png], [scratch_project/results/plots/class_distribution_donut.png] tesislerinde bulunabilir)

## 6. Modelin Kullanımı
Eğitilmiş ve optimize edilmiş en iyi model **RandomForest**, `scratch_project/results/best_optimized_randomforest.pkl` adresine kaydedilmiştir.
Bu modeli yeni veriler üzerinde tahmin yapmak için aşağıdaki gibi yükleyebilir ve kullanabilirsiniz:

```python
import pandas as pd
from utils.model_loader import load_and_predict
from utils.preprocessor import DataPreprocessor

model_path = 'scratch_project/results/best_optimized_randomforest.pkl'
# Yeni verilerinizi yükleyin veya oluşturun
# X_new_raw = pd.DataFrame(...)

# Modeli eğitirken kullanılan aynı ön işleme adımlarını uygulayın (eğer varsa)
# preprocessor = DataPreprocessor(impute_strategy='median', scale_method='standard', detect_outliers='iqr', remove_outliers=True)
# preprocessor.fit(X_train_original_full_dataset, y_train_original_full_dataset) # Modeli eğitmek için kullanılan tüm veriyi kullanarak fit etmelisiniz
# X_new_processed = preprocessor.transform(X_new_raw)

# Tahmin yap
# predictions = load_and_predict(model_path, X_new_processed)
# print(predictions)
```
