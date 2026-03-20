# utils/model_loader.py
# ═══════════════════════════════════════════════════════════
#  Bio-ML Agent — Model Yükleme & Tahmin Utility'si
#
#  Kullanım:
#    from bio_ml_agent.utils.model_loader import load_and_predict, model_info
#
#    # Model bilgisi
#    model_info("results/best_model.pkl")
#
#    # Tahmin yap
#    predictions = load_and_predict("results/best_model.pkl", X_new)
#
#    # Standalone CLI:
#    python utils/model_loader.py --model results/best_model.pkl --data test.csv
# ═══════════════════════════════════════════════════════════

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


def load_model(model_path: str) -> Any:
    """
    Kaydedilmiş bir modeli yükler.

    Args:
        model_path: .pkl dosya yolu

    Returns:
        Yüklenmiş model (sklearn Pipeline veya Estimator)

    Raises:
        FileNotFoundError: Model dosyası bulunamadıysa
    """
    import joblib

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

    model = joblib.load(path)
    return model


def model_info(model_path: str) -> Dict[str, Any]:
    """
    Kaydedilmiş modelin meta bilgilerini döndürür.

    Args:
        model_path: .pkl dosya yolu

    Returns:
        Meta veri dict'i (model adı, görev türü, metrikler vb.)
    """
    path = Path(model_path)
    meta_path = path.parent / path.name.replace(".pkl", "_meta.json")

    info: Dict[str, Any] = {"model_path": str(path)}

    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        info.update(meta)
    else:
        info["warning"] = "Meta veri dosyası bulunamadı"

    # Model tipini kontrol et
    if path.exists():
        model = load_model(str(path))
        info["model_type"] = type(model).__name__

        # Pipeline ise içindeki model adını al
        if hasattr(model, "named_steps"):
            step_names = list(model.named_steps.keys())
            info["pipeline_steps"] = step_names
            # Son adım genelde model
            last_step = model.named_steps[step_names[-1]]
            info["estimator_type"] = type(last_step).__name__

    return info


def load_and_predict(
    model_path: str,
    X: np.ndarray,
    return_proba: bool = False,
) -> np.ndarray:
    """
    Kaydedilmiş modeli yükleyip tahmin yapar.

    Args:
        model_path: .pkl dosya yolu
        X: Tahmin yapılacak özellik matrisi (n_samples, n_features)
        return_proba: True ise olasılık tahmini döndür (sadece sınıflandırma)

    Returns:
        Tahmin sonuçları (np.ndarray)
    """
    model = load_model(model_path)

    if return_proba and hasattr(model, "predict_proba"):
        return model.predict_proba(X)

    return model.predict(X)


def predict_single(
    model_path: str,
    features: dict,
    feature_names: list,
) -> Tuple[Any, Optional[np.ndarray]]:
    """
    Tek bir örnek için tahmin yapar.

    Args:
        model_path: .pkl dosya yolu
        features: {özellik_adı: değer} sözlüğü
        feature_names: Eğitimde kullanılan özellik sırasını belirten liste

    Returns:
        (tahmin, olasılıklar) tuple'ı

    Kullanım:
        prediction, proba = predict_single(
            "results/best_model.pkl",
            {"radius_mean": 14.2, "texture_mean": 19.5, ...},
            feature_names=["radius_mean", "texture_mean", ...]
        )
    """
    # Özellik vektörünü doğru sırayla oluştur
    X = np.array([[features.get(name, 0) for name in feature_names]])

    model = load_model(model_path)
    prediction = model.predict(X)[0]

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]

    return prediction, proba


# ─────────────────────────────────────────────
#  Standalone CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    import pandas as pd

    parser = argparse.ArgumentParser(description="Model Yükleme & Tahmin Aracı")
    subparsers = parser.add_subparsers(dest="command", help="Komut")

    # info komutu
    info_parser = subparsers.add_parser("info", help="Model bilgilerini göster")
    info_parser.add_argument("--model", required=True, help=".pkl dosya yolu")

    # predict komutu
    predict_parser = subparsers.add_parser("predict", help="Tahmin yap")
    predict_parser.add_argument("--model", required=True, help=".pkl dosya yolu")
    predict_parser.add_argument("--data", required=True, help="CSV dosya yolu")
    predict_parser.add_argument("--output", default=None, help="Çıktı CSV yolu")
    predict_parser.add_argument("--sep", default=",", help="CSV ayırıcı")

    args = parser.parse_args()

    if args.command == "info":
        info = model_info(args.model)
        print("\n🧠 Model Bilgileri:")
        print("─" * 40)
        for k, v in info.items():
            print(f"  {k}: {v}")

    elif args.command == "predict":
        print(f"📂 Veri yükleniyor: {args.data}")
        df = pd.read_csv(args.data, sep=args.sep)
        print(f"   Boyut: {df.shape[0]} satır × {df.shape[1]} sütun")

        predictions = load_and_predict(args.model, df.values)
        df["prediction"] = predictions

        if args.output:
            df.to_csv(args.output, index=False)
            print(f"💾 Tahminler kaydedildi: {args.output}")
        else:
            print("\n📊 Tahmin Sonuçları:")
            print(df[["prediction"]].value_counts() if len(df) > 10 else df)

    else:
        parser.print_help()
