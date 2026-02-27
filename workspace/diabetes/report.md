# Meme Kanseri Sınıflandırma Projesi Raporu

## 1. Giriş
Bu proje, Wisconsin Meme Kanseri veri setini kullanarak kötü huylu (malignant) ve iyi huylu (benign) tümörleri sınıflandırmak için bir makine öğrenimi çözümü geliştirmeyi amaçlamaktadır. Proje kapsamında çeşitli sınıflandırma modelleri eğitilmiş, performansları karşılaştırılmış ve elde edilen sonuçlar detaylı görselleştirmelerle desteklenmiştir.

## 2. Veri Seti
Wisconsin Meme Kanseri veri seti, meme dokusunun iğne aspirasyonundan elde edilen sayısal özelliklerden oluşur. Veri seti 569 örnek ve 30 özellik içermektedir. Hedef değişken, tümörün iyi huylu (benign) veya kötü huylu (malignant) olup olmadığını belirten ikili bir kategoridir.
- **Özellikler:** Her bir hücre çekirdeğinin dijital görüntüsünden hesaplanan 30 adet gerçek değerli özellik (örneğin, yarıçap, doku, çevre, alan, pürüzsüzlük, kompaktlık, içbükeylik, içbükey noktalar, simetri, fraktal boyutun ortalaması, standart hatası ve "en kötü" veya en büyük değeri).
- **Hedef:** 0 (Kötü Huylu - Malignant), 1 (İyi Huylu - Benign).

## 3. Metodoloji
Proje aşağıdaki adımları içermiştir:

### 3.1. Veri Yükleme ve Ön İşleme
Veri seti `dataset_catalog` modülünden yüklenmiştir. Veri, eğitim ve test setlerine %80 eğitim, %20 test oranıyla `train_test_split` kullanılarak ayrılmıştır. Stratified sampling, sınıf oranlarının hem eğitim hem de test setlerinde korunmasını sağlamak için kullanılmıştır.

### 3.2. Model Karşılaştırması
Aşağıdaki sınıflandırma modelleri, `StandardScaler` ve `Pipeline` kullanılarak karşılaştırılmıştır:
- Lojistik Regresyon (Logistic Regression)
- Rastgele Orman (Random Forest)
- Gradyan Arttırma (Gradient Boosting)
- Destek Vektör Makinesi (Support Vector Machine - SVC)
- K-En Yakın Komşular (K-Nearest Neighbors - KNN)

Her model, 5 katlı çapraz doğrulama (`StratifiedKFold`) ile eğitilmiş ve değerlendirilmiştir. Performans metrikleri olarak Accuracy, Precision, Recall, F1-score ve ROC AUC kullanılmıştır. Modellerin son performansı, ayrılan test seti üzerinde değerlendirilmiştir.

### 3.3. Görselleştirmeler
En iyi performans gösteren model ve genel veri seti hakkında içgörüler sağlamak için çeşitli görselleştirmeler oluşturulmuştur:
- **Karışıklık Matrisi:** Modelin doğru ve yanlış sınıflandırmalarını gösterir (normal ve normalize edilmiş).
- **ROC Eğrisi:** Modelin farklı karar eşiklerinde doğru pozitif ve yanlış pozitif oranları arasındaki değişimi gösterir.
- **Özellik Önemleri:** Model için en önemli özelliklerin sıralamasını gösterir.
- **Korelasyon Matrisi:** Veri setindeki özellikler arasındaki doğrusal ilişkileri gösterir.
- **Öğrenme Eğrisi:** Eğitim seti boyutu arttıkça modelin eğitim ve çapraz doğrulama skorlarının nasıl değiştiğini gösterir.
- **Sınıf Dağılımı:** Hedef değişkenin sınıflarının dağılımını gösterir.

## 4. Sonuçlar

### 4.1. Model Karşılaştırma Tablosu
Aşağıdaki tablo, modellerin çapraz doğrulama ortalama metriklerini ve test seti performanslarını özetlemektedir:

| Model                  |   CV_Accuracy_Mean |   CV_Precision_Mean |   CV_Recall_Mean |   CV_F1_Mean |   CV_ROC_AUC_Mean |   Test_Accuracy |   Test_Precision |   Test_Recall |   Test_F1 |   Test_ROC_AUC |
|:-----------------------|-------------------:|--------------------:|-----------------:|-------------:|------------------:|----------------:|-----------------:|--------------:|----------:|---------------:|
| Logistic Regression    |           0.98022  |            0.982692 |         0.985965 |     0.984224 |          0.995975 |        0.982456 |         0.986111 |      0.986111 |  0.986111 |       0.995701 |
| Random Forest          |           0.962637 |            0.975519 |         0.964912 |     0.969935 |          0.989577 |        0.95614  |         0.958904 |      0.972222 |  0.965517 |       0.993882 |
| Gradient Boosting      |           0.951648 |            0.961725 |         0.961404 |     0.961363 |          0.991847 |        0.95614  |         0.946667 |      0.986111 |  0.965986 |       0.990741 |
| Support Vector Machine |           0.969231 |            0.97275  |         0.978947 |     0.975615 |          0.995562 |        0.982456 |         0.986111 |      0.986111 |  0.986111 |       0.99504  |
| K-Nearest Neighbors    |           0.962637 |            0.959179 |         0.982456 |     0.970489 |          0.988235 |        0.95614  |         0.958904 |      0.972222 |  0.965517 |       0.978836 |

**En İyi Model (Test ROC AUC'ye Göre):** Logistic Regression

### 4.2. Görselleştirmeler
Projenin `results/plots/` klasöründe aşağıdaki görselleştirmeler bulunmaktadır:
- **Karışıklık Matrisi (Normal ve Normalize Edilmiş):**
  ![Confusion Matrix](plots/confusion_matrix.png)
  Modelin hem iyi huylu hem de kötü huylu vakaları ne kadar iyi sınıflandırdığını gösterir.

- **ROC Eğrisi:**
  ![ROC Curve](plots/roc_curve.png)
  Modelin sınıflandırma eşiğine duyarlılığını ve genel sınıflandırma yeteneğini gösterir. AUC değeri 1'e ne kadar yakınsa, model o kadar iyidir.

- **Özellik Önemleri:**
  ![Feature Importance](plots/feature_importance.png)
  Modelin karar verme sürecinde hangi özelliklerin daha etkili olduğunu vurgular.

- **Veri Seti Korelasyon Matrisi:**
  ![Correlation Matrix](plots/correlation_matrix.png)
  Özellikler arasındaki ilişkileri görselleştirir ve potansiyel çoklu doğrusallık sorunlarına işaret edebilir.

- **Öğrenme Eğrisi:**
  ![Learning Curve](plots/learning_curve.png)
  Modelin eğitim ve çapraz doğrulama skorlarının eğitim veri boyutuyla nasıl değiştiğini gösterir, aşırı uyum veya az uyum sorunlarını teşhis etmeye yardımcı olur.

- **Eğitim Verisi Sınıf Dağılımı:**
  ![Class Distribution](plots/class_distribution.png)
  Hedef sınıfların dengeli olup olmadığını gösterir.

## 5. Tartışma ve Sonuç
Analiz edilen modeller arasında, **Logistic Regression** test setinde en yüksek ROC AUC skorunu elde ederek en iyi performansı göstermiştir. Genel olarak, tüm modellerin yüksek doğruluk ve ROC AUC skorları sergilemesi, veri setinin iyi sınıflandırılabilir olduğunu düşündürmektedir.

Özellik önemleri grafiği, belirli anatomik ve hücresel özelliklerin tümör tipini tahmin etmede kritik olduğunu ortaya koymuştur. Korelasyon matrisi, bazı özellikler arasında güçlü ilişkiler olduğunu göstermiştir, bu da ileri analizlerde özellik seçimi veya boyut indirgeme tekniklerinin faydalı olabileceğini düşündürmektedir.

Öğrenme eğrileri, **Logistic Regression**'ın mevcut veri miktarıyla iyi bir performans sergilediğini ve daha fazla veriyle performansın artıp artmayacağı hakkında bilgi vermektedir.

Bu proje, Meme Kanseri sınıflandırma problemine kapsamlı bir makine öğrenimi yaklaşımı sunmaktadır.

## 6. Gelecek Çalışmalar
- Hiperparametre optimizasyonu (Grid Search, Random Search) ile model performansını daha da iyileştirmek.
- Farklı özellik mühendisliği tekniklerini denemek.
- Anomali tespiti veya transfer öğrenimi gibi daha gelişmiş teknikleri araştırmak.
- Diğer benzer biyomedikal veri setleri üzerinde modelleri test etmek.