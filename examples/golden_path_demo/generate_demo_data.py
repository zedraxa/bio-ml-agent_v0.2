# examples/golden_path_demo/generate_demo_data.py
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
import os

def generate():
    print("------- 🧬 Bio-ML Agent Demo Veri Üretici -------")
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    
    # Hedef isimlerini daha okunabilir yapalım
    df['diagnosis'] = df['target'].map({0: 'malignant', 1: 'benign'})
    df = df.drop(columns=['target'])
    
    output_path = os.path.join(os.path.dirname(__file__), "breast_cancer_data.csv")
    df.to_csv(output_path, index=False)
    print(f"✅ Demo veri seti başarıyla oluşturuldu: {output_path}")
    print(f"📊 Boyut: {df.shape[0]} satır, {df.shape[1]} sütun")
    print("------------------------------------------------")

if __name__ == "__main__":
    generate()
