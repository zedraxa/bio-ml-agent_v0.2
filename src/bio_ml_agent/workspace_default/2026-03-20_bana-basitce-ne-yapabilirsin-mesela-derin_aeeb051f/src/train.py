import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import sys

# Projenin kök dizinini Python path'ine ekleyerek 'utils' modülünün bulunmasını sağlıyoruz.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    # Model karşılaştırma için ajan'ın yerleşik yardımcı fonksiyonunu import ediyoruz.
    from utils.model_compare import compare_models
except ImportError:
    print("HATA: 'utils.model_compare' modülü bulunamadı.")
    print("Lütfen ajanın bu modülü içeren bir ortamda çalıştığından emin olun.")
    sys.exit(1)

def main():
    """
    Ana model eğitim ve karşılaştırma fonksiyonu.
    """
    print("Proje çalışmaya başladı: Meme Kanseri Teşhisi")

    # 1. Veri Yükleme ve Sütun İsimlerini Belirleme
    base_features = [
        'radius', 'texture', 'perimeter', 'area', 'smoothness',
        'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension'
    ]
    suffixes = ['_mean', '_se', '_worst']
    column_names = ['id', 'diagnosis']
    for suffix in suffixes:
        for feature in base_features:
            column_names.append(feature + suffix)

    data_path = 'data/raw/wdbc.data'
    print(f"Veri yükleniyor: {data_path}")
    try:
        df = pd.read_csv(data_path, header=None, names=column_names)
    except FileNotFoundError:
        print(f"HATA: Veri dosyası bulunamadı: {data_path}")
        sys.exit(1)


    # 2. Veri Ön İşleme
    print("Veri ön işleme adımı başlatıldı...")
    df = df.drop('id', axis=1)

    le = LabelEncoder()
    df['diagnosis'] = le.fit_transform(df['diagnosis'])
    print("Hedef değişken 'diagnosis' (Malignant/Benign) sayısal formata dönüştürüldü.")

    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    
    # 3. Veriyi Eğitim ve Test Setlerine Ayırma
    print("Veri, eğitim ve test setlerine ayrılıyor...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Eğitim seti boyutu: {X_train.shape[0]} örnek")
    print(f"Test seti boyutu: {X_test.shape[0]} örnek")

    # 4. Modelleri Karşılaştırma
    print("\nModel karşılaştırma süreci başlatılıyor...")
    print("Standart ölçekleme ve 5-katlı çapraz doğrulama ile 3 model karşılaştırılacak.")
    
    comparator, results_df = compare_models(
        X_train, X_test, y_train, y_test,
        task_type="classification",
        output_dir="results/"
    )
    
    print("\nModel karşılaştırma tamamlandı.")
    print("Sonuçlar 'results/' klasörüne kaydedildi.")
    print("\nEn iyi performans gösteren modelin özeti:")
    print(results_df.iloc[0])
    print("\nProje başarıyla tamamlandı.")

if __name__ == '__main__':
    main()
