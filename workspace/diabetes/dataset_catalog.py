from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine, load_digits
import pandas as pd
import numpy as np
import os

def load_dataset(dataset_name):
    """
    Belirtilen veri setini yükler.
    Katalogdaki scikit-learn otomatik yüklemeli veri setlerini destekler.
    """
    if dataset_name == 'breast_cancer':
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
        feature_names = data.feature_names.tolist()
        return X, y, feature_names
    elif dataset_name == 'diabetes':
        data = load_diabetes()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
        feature_names = data.feature_names.tolist()
        return X, y, feature_names
    elif dataset_name == 'iris':
        data = load_iris()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
        feature_names = data.feature_names.tolist()
        return X, y, feature_names
    elif dataset_name == 'wine':
        data = load_wine()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
        feature_names = data.feature_names.tolist()
        return X, y, feature_names
    elif dataset_name == 'digits':
        data = load_digits()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
        feature_names = data.feature_names.tolist()
        return X, y, feature_names
    # Diğer veri setleri için URL'den indirme veya farklı yükleme mantığı buraya eklenebilir.
    # Şimdilik sadece auto-load olanları destekliyoruz.
    else:
        raise ValueError(f"Bilinmeyen veya desteklenmeyen veri seti: {dataset_name}. Lütfen katalogdan seçim yapın.")