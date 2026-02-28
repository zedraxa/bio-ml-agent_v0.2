import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys

# utils dizinini sys.path'e ekle
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))

from model_compare import compare_models
from visualize import MLVisualizer
from preprocessor import quick_preprocess, analyze_data_quality

def main():
    # Veri yükleme
    data_path = 'data/raw/diabetes.csv'
    if not os.path.exists(data_path):
        print(f"Hata: Veri dosyası bulunamadı: {data_path}")
        return

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Hata: Veri dosyası yüklenirken bir sorun oluştu: {e}")
        return

    print("Veri başarıyla yüklendi.")
    print("Veri setinin ilk 5 satırı:")
    print(df.head())
    print("\nVeri seti bilgisi:")
    df.info()

    # Hedef değişkeni belirle
    target_column = 'Outcome'
    if target_column not in df.columns:
        print(f"Hata: Hedef sütunu '{target_column}' veri setinde bulunamadı.")
        return

    X = df.drop(columns=[target_column])
    y = df[target_column]

    feature_cols = X.columns.tolist()

    # Veri kalitesi analizi
    print("\nVeri Kalitesi Analizi:")
    quality_report = analyze_data_quality(X, feature_names=feature_cols)
    print(quality_report)

    # Veri ön işleme (hızlı ve basit)
    print("\nVeri Ön İşleme...")
    # Sadece sayısal sütunları seç, kategorik sütun yok
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    X_processed = quick_preprocess(X[numeric_cols], scale=True, impute_strategy='median')
    print("Veri ön işleme tamamlandı.")

    # Eğitim ve test setlerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nEğitim seti boyutu: {X_train.shape}")
    print(f"Test seti boyutu: {X_test.shape}")

    # Model karşılaştırması
    print("\nModeller karşılaştırılıyor...")
    best_model, results_dict = compare_models(X_train, X_test, y_train, y_test, task_type="classification", output_dir="results/")
    
    if best_model:
        print(f"\nEn iyi model: {list(results_dict.keys())[list(results_dict.values()).index(max(results_dict.values(), key=lambda x: x.get('roc_auc', -np.inf)))]}")
        
        # Görselleştirmeler
        print("\nGörselleştirmeler oluşturuluyor...")
        viz = MLVisualizer(output_dir="results/plots")
        viz.plot_all(best_model, X_train, X_test, y_train, y_test, 
                     feature_names=feature_cols, df=df, model_name="Best Model", 
                     task_type="classification", target_name=target_column)
        print("Görselleştirmeler tamamlandı ve 'results/plots/' dizinine kaydedildi.")
    else:
        print("En iyi model bulunamadı veya bir hata oluştu.")

if __name__ == "__main__":
    main()