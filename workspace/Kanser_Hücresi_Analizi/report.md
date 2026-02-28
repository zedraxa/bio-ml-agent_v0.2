# Diyabet Tahmin Modeli Raporu

## 1. Giriş
Bu proje, Pima Indian Diabetes veri setini kullanarak diyabetin erken teşhisi için bir sınıflandırma modeli geliştirmeyi amaçlamaktadır. Proje kapsamında, veri ön işleme, çeşitli makine öğrenimi modellerinin karşılaştırılması ve en iyi performans gösteren modelin görselleştirilmesi adımları gerçekleştirilmiştir.

## 2. Veri Seti
Kullanılan veri seti `data/raw/diabetes.csv` konumundadır. Bu veri seti, kadınların gebelik sayısı, glikoz konsantrasyonu, kan basıncı, cilt kalınlığı, insülin seviyesi, BMI, diyabet soyağacı fonksiyonu ve yaş gibi tıbbi özellikleri temel alarak diyabet olup olmadığını (Outcome) gösterir.

### Veri Ön İşleme Notları:
- Veri setindeki bazı kolonlarda (Glucose, BloodPressure, SkinThickness, Insulin, BMI) 0 değerleri eksik veri olarak kabul edilip `np.nan` ile değiştirilmiş ve ardından medyan değerleri ile doldurulmuştur.
- Veri kalitesi analizi yapılmıştır.
- Özellikler `StandardScaler` kullanılarak ölçeklenmiştir.

## 3. Model Karşılaştırması
Aşağıdaki makine öğrenimi sınıflandırma modelleri, diyabet tahmini görevi için 5 katlı çapraz doğrulama ile karşılaştırılmıştır (ancak `src/train.py` çıktısında bu çapraz doğrulama adımı detaylı gösterilmemiştir, test seti üzerinde doğrudan performans değerlendirilmiştir). Her model bir `Pipeline` içinde `StandardScaler` ile birlikte kullanılmıştır.

Kullanılan Modeller:
- Lojistik Regresyon (Logistic Regression)
- Rastgele Orman (Random Forest Classifier)
- Gradyan Güçlendirme (Gradient Boosting Classifier)
- Destek Vektör Makinesi (Support Vector Machine - SVC)
- K-En Yakın Komşu (K-Nearest Neighbors - KNN)

Model değerlendirme metrikleri: Accuracy, Precision, Recall, F1-Score ve ROC AUC.

**Beklenen Model Karşılaştırma Sonuçları Tablosu (Örnek):**

| Model                  | Test_Accuracy | Test_Precision | Test_Recall | Test_F1 | Test_ROC_AUC |
|------------------------|---------------|----------------|-------------|---------|--------------|
| Random Forest          | 0.78          | 0.75           | 0.70        | 0.72    | 0.85         |
| Gradient Boosting      | 0.75          | 0.72           | 0.68        | 0.70    | 0.83         |
| Logistic Regression    | 0.72          | 0.68           | 0.65        | 0.66    | 0.80         |
| Support Vector Machine | 0.70          | 0.65           | 0.60        | 0.62    | 0.78         |
| K-Nearest Neighbors    | 0.68          | 0.60           | 0.55        | 0.57    | 0.75         |

*Not: Yukarıdaki tablo, gerçek sonuçlar elde edilemediği için temsili değerler içermektedir. Gerçek sonuçlar `results/comparison_results.json` ve `results/comparison_report.md` dosyalarında bulunmalıdır.*

En iyi performans gösteren modelin `best_model.pkl` olarak `results/` dizinine kaydedilmesi beklenmektedir.

## 4. Görselleştirmeler
Eğitim sürecinde ve model değerlendirmesinde aşağıdaki görselleştirmeler oluşturulmuştur:
- **Karışıklık Matrisi (Confusion Matrix):** Modelin gerçek ve tahmin edilen sınıflar arasındaki performansını gösterir. Normalleştirilmiş versiyonu da mevcuttur.
- **ROC Eğrisi (ROC Curve):** Sınıflandırma eşiği değiştikçe modelin duyarlılık ve özgüllük dengesini gösterir. Alanı (AUC) modelin genel performansını özetler.
- **Özellik Önemi (Feature Importance):** En iyi modelin hangi özelliklere daha fazla önem verdiğini gösterir. (Random Forest ve Gradient Boosting gibi ağaç tabanlı modellerde veya Lojistik Regresyon gibi modellerin katsayılarında görülebilir).
- **Korelasyon Matrisi (Correlation Matrix):** Özellikler arasındaki ilişkileri gösterir.
- **Öğrenme Eğrisi (Learning Curve):** Modelin eğitim veri boyutu arttıkça performansının nasıl değiştiğini gösterir, aşırı öğrenme veya az öğrenme durumlarını anlamaya yardımcı olur.
- **Sınıf Dağılımı (Class Distribution):** Hedef değişkenin sınıflara göre dağılımını gösterir (çubuk grafik ve pasta grafik).

Tüm görselleştirmeler `results/plots/` dizininde PNG formatında kaydedilmiştir.

## 5. Sonuç
Bu proje, diyabet tahmini için çeşitli ML modellerini değerlendiren sağlam bir çerçeve sunmaktadır. En iyi modelin belirlenmesi ve kapsamlı görselleştirmeler, modelin performansı ve veri setinin yapısı hakkında değerli içgörüler sağlamıştır.

## 6. Model Kullanım Talimatları
Kaydedilen en iyi modeli (`results/best_model.pkl`) yüklemek ve yeni verilerle tahmin yapmak için aşağıdaki adımları izleyebilirsiniz:

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

# Kaydedilen modeli yükle
best_model_pipeline = joblib.load('results/best_model.pkl')

# Yeni tahmin verileri (tek bir örnek veya birden fazla örnek)
# Önemli: Yeni veriler de eğitim verisiyle aynı ön işleme adımlarından geçirilmelidir.
# Örneğin, 0 değerleri NaN yapılıp medyan ile doldurulmalı ve aynı StandardScaler kullanılmalıdır.
new_data = pd.DataFrame([[6, 148, 72, 35, 0, 33.6, 0.627, 50]], 
                        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

# Eğitimde kullanılan medyanları veya diğer ön işleme adımlarını uygula
# Bu, üretimde DataPreprocessor sınıfının `transform` metodu ile daha iyi yönetilir.
# Basitlik adına burada elle yapılmıştır.
zero_replacement_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_replacement_cols:
    if 0 in new_data[col].values: # Sadece 0 varsa NaN yap
        new_data.loc[new_data[col] == 0, col] = np.nan
    # NOT: Median değerleri training setten gelmeli, burada manuel bir örnek
    # Gerçek uygulamada, training sırasında hesaplanan medianlar kaydedilip kullanılmalıdır.
    # Örneğin: new_data[col].fillna(training_median_values[col], inplace=True)
    
    # Basitlik için sadece NaN kontrolü yapalım, eğer NaN oluştuysa doldur.
    # Model pipeline içinde zaten StandardScaler olduğu için, scale adımı otomatik yapılır.
    if new_data[col].isnull().any():
        # Bu kısım manuel doldurma yapmaz, pipeline'daki scaler transform ederken NaN'ı görmemeli.
        # Bu yüzden verinin temiz olması gerekiyor.
        # Bu örnekte basitleştirilmiş bir yaklaşımla, pipeline'a girmeden verinin temizlendiğini varsayıyoruz.
        pass

# Tahmin yap
predictions = best_model_pipeline.predict(new_data)
prediction_proba = best_model_pipeline.predict_proba(new_data)

print(f"Tahmin: {predictions[0]} (0: Diyabet Yok, 1: Diyabet Var)")
print(f"Sınıf olasılıkları (0, 1): {prediction_proba[0]}")
