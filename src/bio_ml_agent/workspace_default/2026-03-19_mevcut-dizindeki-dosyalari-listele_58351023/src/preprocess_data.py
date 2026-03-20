import pandas as pd
import numpy as np

def preprocess_heart_disease_data():
    """
    Loads the raw Cleveland heart disease dataset, preprocesses it, and saves
    the clean version.
    """
    # Sütun isimleri UCI veri seti açıklamasından alınmıştır
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    # Veri setini yükle
    input_path = 'data/raw/heart.csv'
    output_path = 'data/processed/processed_heart.csv'
    
    df = pd.read_csv(input_path, header=None, names=column_names, na_values='?')
    
    print("Veri setinin ilk 5 satırı (ön işleme öncesi):")
    print(df.head())
    
    # Eksik değerleri medyan ile doldurma
    for col in df.columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"'{col}' sütunundaki eksik değerler medyan ({median_val}) ile dolduruldu.")

    # Hedef değişkenini ikili hale getirme (0: hastalık yok, 1: hastalık var)
    # Orjinal veride 0 sağlıklı, 1,2,3,4 ise hasta anlamına gelir.
    df['target'] = (df['target'] > 0).astype(int)
    
    print("\nVeri setinin ilk 5 satırı (ön işleme sonrası):")
    print(df.head())
    
    # İşlenmiş veriyi kaydet
    df.to_csv(output_path, index=False)
    print(f"\nİşlenmiş veri başarıyla '{output_path}' adresine kaydedildi.")

if __name__ == "__main__":
    preprocess_heart_disease_data()