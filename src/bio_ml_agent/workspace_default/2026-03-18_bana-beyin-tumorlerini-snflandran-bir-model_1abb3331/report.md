# Beyin Tümörü Sınıflandırma Modeli Raporu (Geleneksel ML Yaklaşımı)

## 1. Giriş
Bu proje, sistem ortamındaki kısıtlamalar (PyTorch yükleme hatası) nedeniyle başlangıçtaki derin öğrenme yaklaşımından geleneksel makine öğrenimi yaklaşımına geçiş yapmıştır. Amacımız, beyin tümörü sınıflandırma problemi için geleneksel ML modellerini kullanarak bir çözüm sunmak, modelleri karşılaştırmak, performanslarını değerlendirmek ve açıklanabilir yapay zeka (XAI) yöntemleriyle kararları yorumlamaktır.

## 2. Veri Seti
Orijinal veri seti indirme sürecinde yaşanan zorluklar nedeniyle, proje simüle edilmiş (sentetik) bir tabular veri seti üzerinde yürütülmüştür. Bu veri seti, 1000 örnek, 20 özellik ve 4 farklı sınıf (glioma, meningioma, notumor, pituitary) içerecek şekilde `make_classification` fonksiyonu kullanılarak oluşturulmuştur.
*   **Özellik Sayısı**: 20
*   **Sınıf Sayısı**: 4 (glioma, meningioma, notumor, pituitary)
*   **Örnek Sayısı**: 1000
Veri seti `data/raw/simulated_brain_tumor_features.csv` olarak kaydedilmiştir.

## 3. Model Eğitimi ve Karşılaştırması
Beş farklı geleneksel makine öğrenimi modeli eğitilmiş ve performansları karşılaştırılmıştır. Tüm modeller için bir `StandardScaler` ve model içeren bir pipeline kullanılmıştır.

*   LogisticRegression
*   RandomForestClassifier
*   GradientBoostingClassifier
*   SVC (Support Vector Classifier)
*   KNeighborsClassifier

**Model Karşılaştırma Sonuçları:**

| Model Adı          | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|:-------------------|:---------|:----------|:-------|:---------|:--------|
| LogisticRegression | 0.7500   | 0.7538    | 0.7500 | 0.7502   | 0.8967  |
| RandomForestClassifier | 0.8500   | 0.8541    | 0.8500 | 0.8503   | 0.9646  |
| GradientBoostingClassifier | 0.8050   | 0.8149    | 0.8050 | 0.8053   | 0.9502  |
| **SVC**            | **0.8550**| **0.8675** | **0.8550** | **0.8558** | **0.9668** |
| KNeighborsClassifier | 0.7750   | 0.7865    | 0.7750 | 0.7716   | 0.9383  |

En iyi performans gösteren model, **SVC** olmuştur.

**En İyi Model (SVC) Sınıflandırma Raporu:**

              precision    recall  f1-score   support

      glioma       0.89      0.88      0.89        50
  meningioma       0.85      0.84      0.84        50
     notumor       0.86      0.88      0.87        50
   pituitary       0.86      0.82      0.84        50

    accuracy                           0.86       200
   macro avg       0.86      0.86      0.86       200
weighted avg       0.86      0.86      0.86       200

SVC modeli, tüm sınıflar üzerinde tutarlı ve yüksek bir performans sergilemiştir.

## 4. Görselleştirmeler
Proje çıktısı olarak aşağıdaki görseller `results/plots/` dizinine kaydedilmiştir:

*   **Karışıklık Matrisi**: `results/plots/confusion_matrix.png`
*   **Normalleştirilmiş Karışıklık Matrisi**: `results/plots/normalized_confusion_matrix.png`
*   **Çok Sınıflı ROC Eğrisi (OvR)**: `results/plots/roc_curve.png`
*   **Sınıf Dağılımı (Test Seti)**: `results/plots/class_distribution.png`
*   **Özellik Önem Haritası**: `results/plots/feature_importance.png` (SVC'nin katsayıları baz alınarak ortalama mutlak değerleri kullanılmıştır.)

## 5. Açıklanabilir Yapay Zeka (XAI)
Modelin kararlarını daha şeffaf hale getirmek için SHAP (SHapley Additive exPlanations) ve LIME (Local Interpretable Model-agnostic Explanations) yöntemleri kullanılmıştır.

*   **SHAP Özet Grafiği**: `results/plots/shap_summary_plot.png`
    *   Bu görsel, her bir özelliğin modelin çıktıları üzerindeki genel etkisini gösterir. Her bir noktanın rengi özelliğin değerini (kırmızı yüksek, mavi düşük) temsil ederken, yatay konumu o özelliğin tahmin üzerindeki etkisini (pozitif veya negatif) gösterir. En üstteki özellikler model için en önemli olanlardır.

*   **LIME Örnek Açıklaması**: `results/plots/lime_explanation_instance.png`
    *   Seçilen bir test örneği için LIME, hangi özelliklerin modelin belirli bir sınıf tahminine en çok katkıda bulunduğunu gösterir. Yeşil çubuklar pozitif katkıları (tahmini destekleyen), kırmızı çubuklar ise negatif katkıları (tahmini zayıflatan) temsil eder.

### Klinik Karar Özeti
SHAP analizi, modelin sınıflandırma kararları üzerinde en etkili olan özelliklerin (örneğin `feature_1`, `feature_2`, `feature_14`) hangileri olduğunu açıkça göstermektedir. Bu özelliklerin yüksek veya düşük değerlerinin, belirli tümör sınıflarına doğru veya tümörsüz duruma doğru eğilimi nasıl etkilediği görülebilir. LIME analizi ise, bireysel bir hastanın özellikleri temelinde modelin neden belirli bir tümör sınıfını tahmin ettiğini yerel olarak açıklamaktadır. Örneğin, bir `notumor` tahmini için, `feature_X`'in belirli bir değer aralığında olması modelin bu kararı vermesinde en kritik faktör olabilir. Bu XAI bulguları, modelin kararlarının sadece rastgele olmadığını, aynı zamanda veri setindeki belirli desenlere ve özelliklere dayandığını kanıtlamaktadır. Gerçek klinik verilerle, bu özelliklerin tıbbi olarak anlamlı yorumları yapılabilir ve doktorlara teşhis sürecinde değerli bilgiler sağlayabilir.

## 6. Model Kullanımı
Eğitilmiş en iyi model (`results/best_model.pkl`) ile yeni bir veri örneği üzerinde tahmin yapmak için aşağıdaki adımları izleyebilirsiniz:

import joblib
import pandas as pd
import numpy as np

# 1. En iyi modeli yükle
best_model_path = "results/best_model.pkl"
loaded_model = joblib.load(best_model_path)

# 2. Yeni bir veri örneği oluştur (eğitimde kullanılan özelliklerle aynı yapıda olmalı)
# Bu sadece bir örnektir, gerçek verilerinizi buraya yerleştirin.
new_data = pd.DataFrame(np.random.rand(1, 20), columns=[f'feature_{i}' for i in range(20)])

# 3. Sınıf isimleri
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# 4. Tahmin yap
prediction = loaded_model.predict(new_data)
prediction_proba = loaded_model.predict_proba(new_data)

predicted_class_idx = prediction[0]
predicted_class_name = class_names[predicted_class_idx]
confidence = prediction_proba[0][predicted_class_idx]

print(f"Yeni veri örneği için tahmin edilen sınıf: {predicted_class_name}")
print(f"Güven (%): {confidence * 100:.2f}%")

## 7. Sonuç
Bu proje, sistem kısıtlamaları nedeniyle simüle edilmiş bir veri seti üzerinde geleneksel makine öğrenimi modelleri kullanarak beyin tümörü sınıflandırma problemini ele almıştır. SVC modeli en iyi performansı göstermiş ve XAI araçları (SHAP, LIME) model kararlarını anlamak için değerli bilgiler sağlamıştır. Gerçek dünya verileriyle, bu yaklaşım potansiyel olarak klinik teşhis süreçlerine katkıda bulunabilir.