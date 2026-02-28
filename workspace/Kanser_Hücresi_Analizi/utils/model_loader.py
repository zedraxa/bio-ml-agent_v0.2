import joblib
import pandas as pd
import numpy as np
import os

def load_model(model_path):
    """
    Belirtilen yoldan bir modeli yükler.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
    
    try:
        model = joblib.load(model_path)
        print(f"Model başarıyla yüklendi: {model_path}")
        return model
    except Exception as e:
        print(f"Model yüklenirken hata oluştu: {e}")
        return None

def load_and_predict(model_path, X_new):
    """
    Bir modeli yükler ve yeni veriler üzerinde tahminler yapar.
    
    Args:
        model_path (str): Yüklenecek modelin yolu (.pkl dosyası).
        X_new (pd.DataFrame or np.array): Tahminlerin yapılacağı yeni veri.
        
    Returns:
        np.array: Modelin tahminleri.
    """
    model = load_model(model_path)
    if model is None:
        return None
    
    try:
        predictions = model.predict(X_new)
        print("Tahminler başarıyla yapıldı.")
        return predictions
    except Exception as e:
        print(f"Tahmin yaparken hata oluştu: {e}")
        return None

def load_and_predict_proba(model_path, X_new):
    """
    Bir modeli yükler ve yeni veriler üzerinde olasılık tahminleri yapar.
    (Sadece sınıflandırma modelleri için geçerlidir.)
    
    Args:
        model_path (str): Yüklenecek modelin yolu (.pkl dosyası).
        X_new (pd.DataFrame or np.array): Olasılık tahminlerinin yapılacağı yeni veri.
        
    Returns:
        np.array: Modelin olasılık tahminleri.
    """
    model = load_model(model_path)
    if model is None:
        return None
    
    if not hasattr(model, 'predict_proba'):
        print("Yüklenen modelde predict_proba metodu bulunmuyor.")
        return None

    try:
        probabilities = model.predict_proba(X_new)
        print("Olasılık tahminleri başarıyla yapıldı.")
        return probabilities
    except Exception as e:
        print(f"Olasılık tahmini yaparken hata oluştu: {e}")
        return None