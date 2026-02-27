# Meme Kanseri Sınıflandırma Projesi

Bu proje, Wisconsin Meme Kanseri veri setini kullanarak kötü huylu (malignant) ve iyi huylu (benign) tümörleri sınıflandırmak için makine öğrenimi modellerini eğitir ve karşılaştırır.

## Proje Hedefleri
- Wisconsin Meme Kanseri veri setini yüklemek ve ön işlemek.
- En az 3 farklı sınıflandırma modelini (`LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`, `SVC`, `KNeighborsClassifier`) `StandardScaler` ve `Pipeline` kullanarak eğitmek ve karşılaştırmak.
- Modellerin performansını (Accuracy, Precision, Recall, F1-score, ROC AUC) 5 katlı çapraz doğrulama ve test seti üzerinde değerlendirmek.
- Çeşitli grafikler (Karışıklık Matrisi, ROC Eğrisi, Özellik Önemleri, Korelasyon Matrisi, Öğrenme Eğrisi, Sınıf Dağılımı) ile analiz sonuçlarını görselleştirmek.
- Proje adımlarını ve sonuçlarını detaylı bir rapor (`report.md`) halinde sunmak.

## Kurulum
1. Bu projeyi klonlayın:

   git clone <repo-url>
   cd <proje-adi>

2. Gerekli Python kütüphanelerini yükleyin:

   pip install -r requirements.txt

## Kullanım
Proje ana betiğini çalıştırmak için:

python src/train.py

Bu komut:
1. `breast_cancer` veri setini yükler.
2. Veriyi eğitim ve test setlerine ayırır.
3. Belirtilen modelleri eğitir ve test eder, sonuçları `results/comparison_results.json` ve `results/comparison_report.md` dosyalarına kaydeder.
4. En iyi performans gösteren modelin performansını görselleştirir ve çıktıları `results/plots/` klasörüne kaydeder.

## Çıktılar
Proje çalıştırıldıktan sonra aşağıdaki çıktılar `results/` klasöründe bulunacaktır:
- `comparison_results.json`: Modellerin karşılaştırma metriklerini içeren JSON dosyası.
- `comparison_report.md`: Modellerin performansını özetleyen Markdown raporu.
- `plots/`: Oluşturulan tüm grafiklerin PNG formatında kaydedildiği klasör.
  - `confusion_matrix.png`
  - `roc_curve.png`
  - `feature_importance.png`
  - `correlation_matrix.png`
  - `learning_curve.png`
  - `class_distribution.png`
- `report.md`: Projenin genel bir özetini, metodolojisini ve bulgularını içeren detaylı rapor.

## Veri Seti
Wisconsin Meme Kanseri (Diagnostic) Veri Seti, scikit-learn kütüphanesi aracılığıyla erişilebilir. Her örnek, meme dokusunun iğne aspirasyonundan elde edilen 30 adet gerçek değerli özellik içerir ve hedef değişken iki sınıflıdır: kötü huylu (malignant) veya iyi huylu (benign).