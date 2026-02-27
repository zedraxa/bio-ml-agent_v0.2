# utils/visualize.py
# ═══════════════════════════════════════════════════════════
#  Bio-ML Agent — Görselleştirme Modülü
#
#  Desteklenen Grafikler:
#    1. Confusion Matrix
#    2. ROC Curve (binary & multi-class)
#    3. Feature Importance
#    4. Korelasyon Matrisi (Heatmap)
#    5. Learning Curve
#    6. Model Karşılaştırma (bar chart)
#    7. Sınıf Dağılımı
#
#  Kullanım:
#    from utils.visualize import MLVisualizer
#    viz = MLVisualizer(output_dir="results/plots")
#    viz.plot_all(model, X_train, X_test, y_train, y_test, feature_names, df)
#
#  Veya tekil fonksiyonlar:
#    from utils.visualize import plot_confusion_matrix, plot_roc_curve
# ═══════════════════════════════════════════════════════════

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# matplotlib backend — display gerektirmeyen Agg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.base import BaseEstimator
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    auc,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  Ortak Stil Yapılandırması
# ─────────────────────────────────────────────

# Modern, premium renk paleti
COLORS = {
    "primary": "#6366F1",       # Indigo
    "secondary": "#8B5CF6",     # Violet
    "success": "#10B981",       # Emerald
    "warning": "#F59E0B",       # Amber
    "danger": "#EF4444",        # Red
    "info": "#06B6D4",          # Cyan
    "neutral": "#64748B",       # Slate
}

# Çoklu model renkleri
MODEL_COLORS = [
    "#6366F1", "#10B981", "#F59E0B", "#EF4444",
    "#8B5CF6", "#06B6D4", "#EC4899", "#14B8A6",
    "#F97316", "#3B82F6", "#84CC16", "#A855F7",
]

# Heatmap renk paleti
HEATMAP_CMAP = "RdYlBu_r"


def _apply_style():
    """Premium modern stil uygular."""
    plt.rcParams.update({
        "figure.facecolor": "#FAFAFA",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#E2E8F0",
        "axes.labelcolor": "#334155",
        "axes.titlecolor": "#1E293B",
        "axes.grid": True,
        "grid.color": "#F1F5F9",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "text.color": "#334155",
        "xtick.color": "#64748B",
        "ytick.color": "#64748B",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.titlesize": 16,
    })
    # Google Fonts kullanılamasa da sans-serif ayarla
    try:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    except Exception:
        pass


def _save_figure(fig: plt.Figure, path: Path, dpi: int = 150) -> Path:
    """Figürü kaydet ve kapat."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  📊 Grafik kaydedildi: {path}")
    return path


# ═══════════════════════════════════════════════════════════
#  1. CONFUSION MATRIX
# ═══════════════════════════════════════════════════════════

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Union[str, Path] = "results/confusion_matrix.png",
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    normalize: bool = False,
) -> Path:
    """
    Confusion matrix grafiği oluşturur.

    Args:
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
        output_path: Çıktı dosya yolu
        labels: Sınıf isimleri
        title: Grafik başlığı
        normalize: Normalize edilsin mi ('true' = satır bazlı)

    Returns:
        Kaydedilen dosya yolu
    """
    _apply_style()
    path = Path(output_path)

    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]

    fig_size = max(6, n_classes * 0.8 + 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    if normalize:
        # Satır bazlı normalize (her sınıf için recall)
        cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        display_cm = cm_norm
        fmt = ".2f"
        title_suffix = " (Normalize)"
    else:
        display_cm = cm
        fmt = "d"
        title_suffix = ""

    # Seaborn benzeri heatmap
    im = ax.imshow(display_cm, interpolation="nearest", cmap="Blues", aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    # Hücre değerlerini yaz
    thresh = display_cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = f"{display_cm[i, j]:{fmt}}"
            color = "white" if display_cm[i, j] > thresh else "#334155"
            ax.text(j, i, val, ha="center", va="center", color=color,
                    fontsize=10, fontweight="bold")

    # Etiketler
    tick_labels = labels if labels else [str(c) for c in sorted(np.unique(np.concatenate([y_true, y_pred])))]
    ax.set_xticks(range(len(tick_labels)))
    ax.set_yticks(range(len(tick_labels)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Tahmin Edilen", fontweight="bold")
    ax.set_ylabel("Gerçek", fontweight="bold")
    ax.set_title(f"🔲 {title}{title_suffix}", fontweight="bold", pad=15)

    fig.tight_layout()
    return _save_figure(fig, path)


# ═══════════════════════════════════════════════════════════
#  2. ROC CURVE
# ═══════════════════════════════════════════════════════════

def plot_roc_curve(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_path: Union[str, Path] = "results/roc_curve.png",
    title: str = "ROC Eğrisi",
    labels: Optional[List[str]] = None,
) -> Path:
    """
    ROC eğrisi grafiği oluşturur. Binary ve multi-class destekler.

    Args:
        model: Eğitilmiş model (predict_proba gerekli)
        X_test: Test özellikleri
        y_test: Test etiketleri
        output_path: Çıktı dosya yolu
        title: Grafik başlığı
        labels: Sınıf isimleri

    Returns:
        Kaydedilen dosya yolu
    """
    _apply_style()
    path = Path(output_path)

    if not hasattr(model, "predict_proba"):
        print(f"  ⚠️ Model predict_proba desteklemiyor, ROC eğrisi oluşturulamadı.")
        return path

    classes = np.unique(y_test)
    n_classes = len(classes)

    fig, ax = plt.subplots(figsize=(8, 7))

    if n_classes == 2:
        # Binary classification
        y_score = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=classes[1])
        roc_auc = auc(fpr, tpr)

        label_name = labels[1] if labels and len(labels) > 1 else f"Sınıf {classes[1]}"
        ax.plot(fpr, tpr, color=COLORS["primary"], lw=2.5,
                label=f"{label_name} (AUC = {roc_auc:.4f})")
        ax.fill_between(fpr, tpr, alpha=0.15, color=COLORS["primary"])
    else:
        # Multi-class — One-vs-Rest
        y_bin = label_binarize(y_test, classes=classes)
        y_score = model.predict_proba(X_test)

        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            color = MODEL_COLORS[i % len(MODEL_COLORS)]
            label_name = labels[i] if labels and i < len(labels) else f"Sınıf {cls}"
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f"{label_name} (AUC = {roc_auc:.3f})")

    # Diyagonal çizgi
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.4, label="Rastgele (AUC = 0.5)")

    ax.set_xlabel("Yanlış Pozitif Oranı (FPR)", fontweight="bold")
    ax.set_ylabel("Doğru Pozitif Oranı (TPR)", fontweight="bold")
    ax.set_title(f"📈 {title}", fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    fig.tight_layout()
    return _save_figure(fig, path)


# ═══════════════════════════════════════════════════════════
#  3. FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════

def plot_feature_importance(
    model: BaseEstimator,
    feature_names: List[str],
    output_path: Union[str, Path] = "results/feature_importance.png",
    title: str = "Özellik Önemliliği",
    top_n: int = 20,
) -> Path:
    """
    Özellik önemliliği grafiği oluşturur.
    Tree-based modeller (feature_importances_) ve linear modeller (coef_) desteklenir.

    Args:
        model: Eğitilmiş model
        feature_names: Özellik isimleri
        output_path: Çıktı dosya yolu
        title: Grafik başlığı
        top_n: Gösterilecek en önemli N özellik

    Returns:
        Kaydedilen dosya yolu
    """
    _apply_style()
    path = Path(output_path)

    # Pipeline'dan son adımı çıkar
    estimator = model
    if hasattr(model, "named_steps"):
        # Pipeline — son step'i al
        step_names = list(model.named_steps.keys())
        estimator = model.named_steps[step_names[-1]]
    elif hasattr(model, "steps"):
        estimator = model.steps[-1][1]

    # Importance değerlerini al
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
        importance_type = "Gini Importance"
    elif hasattr(estimator, "coef_"):
        coef = estimator.coef_
        if coef.ndim > 1:
            importances = np.abs(coef).mean(axis=0)
        else:
            importances = np.abs(coef.ravel())
        importance_type = "|Katsayı| (Ortalama)"
    else:
        print(f"  ⚠️ Model özellik önemliliği desteklemiyor.")
        # Boş grafik oluştur
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Bu model özellik önemliliği desteklemiyor",
                ha="center", va="center", fontsize=14, color=COLORS["neutral"])
        ax.set_axis_off()
        return _save_figure(fig, path)

    # Sırala ve ilk N'i al
    n_features = min(top_n, len(feature_names), len(importances))
    indices = np.argsort(importances)[::-1][:n_features]

    sorted_names = [feature_names[i] for i in indices]
    sorted_values = importances[indices]

    # Grafik — yatay bar
    fig_height = max(5, n_features * 0.4 + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    # Renk gradyanı
    norm_values = sorted_values / (sorted_values.max() + 1e-10)
    colors = plt.cm.viridis(norm_values * 0.8 + 0.2)

    bars = ax.barh(range(n_features), sorted_values[::-1], color=colors[::-1],
                   edgecolor="white", linewidth=0.5, height=0.7)

    # Değer etiketleri
    for bar, val in zip(bars, sorted_values[::-1]):
        ax.text(bar.get_width() + sorted_values.max() * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9, color=COLORS["neutral"])

    ax.set_yticks(range(n_features))
    ax.set_yticklabels(sorted_names[::-1], fontsize=10)
    ax.set_xlabel(importance_type, fontweight="bold")
    ax.set_title(f"🎯 {title} (Top {n_features})", fontweight="bold", pad=15)

    fig.tight_layout()
    return _save_figure(fig, path)


# ═══════════════════════════════════════════════════════════
#  4. KORELASYON MATRİSİ
# ═══════════════════════════════════════════════════════════

def plot_correlation_matrix(
    df: pd.DataFrame,
    output_path: Union[str, Path] = "results/correlation_matrix.png",
    title: str = "Korelasyon Matrisi",
    method: str = "pearson",
    figsize: Optional[Tuple[int, int]] = None,
    annot_threshold: int = 15,
) -> Path:
    """
    Korelasyon matrisi heatmap'i oluşturur.

    Args:
        df: Sayısal sütunları olan DataFrame
        output_path: Çıktı dosya yolu
        title: Grafik başlığı
        method: Korelasyon metodu ('pearson', 'spearman', 'kendall')
        figsize: Grafik boyutu (None ise otomatik)
        annot_threshold: Bu sayıdan az sütun varsa hücre değerlerini göster

    Returns:
        Kaydedilen dosya yolu
    """
    _apply_style()
    path = Path(output_path)

    # Sadece sayısal sütunlar
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr(method=method)

    n_cols = len(corr.columns)
    if figsize is None:
        size = max(8, n_cols * 0.6 + 2)
        figsize = (size, size)

    fig, ax = plt.subplots(figsize=figsize)

    # Mask — üst üçgen
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    # Heatmap
    im = ax.imshow(
        np.ma.array(corr.values, mask=mask),
        cmap=HEATMAP_CMAP,
        vmin=-1, vmax=1,
        aspect="auto",
        interpolation="nearest",
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Korelasyon", fontsize=10)

    # Hücre değerleri (küçük matrisler için)
    show_annot = n_cols <= annot_threshold
    if show_annot:
        for i in range(n_cols):
            for j in range(n_cols):
                if not mask[i, j]:
                    val = corr.values[i, j]
                    color = "white" if abs(val) > 0.7 else "#334155"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=max(7, 11 - n_cols // 3), color=color, fontweight="bold")

    # Etiketler
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_cols))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=max(7, 10 - n_cols // 5))
    ax.set_yticklabels(corr.columns, fontsize=max(7, 10 - n_cols // 5))
    ax.set_title(f"🔥 {title} ({method.capitalize()})", fontweight="bold", pad=15)

    fig.tight_layout()
    return _save_figure(fig, path)


# ═══════════════════════════════════════════════════════════
#  5. LEARNING CURVE
# ═══════════════════════════════════════════════════════════

def plot_learning_curve(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    output_path: Union[str, Path] = "results/learning_curve.png",
    title: str = "Öğrenme Eğrisi",
    cv: int = 5,
    scoring: str = "accuracy",
    n_points: int = 10,
) -> Path:
    """
    Öğrenme eğrisi (train vs validation skor) grafiği oluşturur.

    Args:
        model: Model (clone edilecek)
        X: Özellikler
        y: Etiketler
        output_path: Çıktı dosya yolu
        title: Grafik başlığı
        cv: Çapraz doğrulama fold sayısı
        scoring: Metrik ('accuracy', 'f1', 'r2', vs.)
        n_points: Eğrideki nokta sayısı

    Returns:
        Kaydedilen dosya yolu
    """
    _apply_style()
    path = Path(output_path)

    train_sizes = np.linspace(0.1, 1.0, n_points)

    try:
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y,
            train_sizes=train_sizes,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
        )
    except Exception as e:
        print(f"  ⚠️ Learning curve hesaplanamadı: {e}")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, f"Learning curve hesaplanamadı:\n{e}",
                ha="center", va="center", fontsize=11, color=COLORS["danger"])
        ax.set_axis_off()
        return _save_figure(fig, path)

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Eğitim eğrisi
    ax.plot(train_sizes_abs, train_mean, "o-", color=COLORS["primary"],
            lw=2.5, markersize=6, label=f"Eğitim {scoring}")
    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color=COLORS["primary"])

    # Doğrulama eğrisi
    ax.plot(train_sizes_abs, val_mean, "s-", color=COLORS["success"],
            lw=2.5, markersize=6, label=f"Doğrulama {scoring}")
    ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color=COLORS["success"])

    # Boşluk göstergesi (overfitting/underfitting)
    gap = train_mean[-1] - val_mean[-1]
    if gap > 0.1:
        status = "⚠️ Overfitting riski var"
        status_color = COLORS["warning"]
    elif val_mean[-1] < 0.6:
        status = "⚠️ Underfitting riski var"
        status_color = COLORS["danger"]
    else:
        status = "✅ İyi genelleme"
        status_color = COLORS["success"]

    ax.text(0.98, 0.02, f"Boşluk: {gap:.4f} — {status}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, color=status_color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=status_color, alpha=0.9))

    ax.set_xlabel("Eğitim Örnekleri Sayısı", fontweight="bold")
    ax.set_ylabel(scoring.capitalize(), fontweight="bold")
    ax.set_title(f"📉 {title}", fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)

    fig.tight_layout()
    return _save_figure(fig, path)


# ═══════════════════════════════════════════════════════════
#  6. MODEL KARŞILAŞTIRMA (Bar Chart)
# ═══════════════════════════════════════════════════════════

def plot_model_comparison(
    model_names: List[str],
    metrics_dict: Dict[str, List[float]],
    output_path: Union[str, Path] = "results/model_comparison.png",
    title: str = "Model Karşılaştırması",
) -> Path:
    """
    Birden fazla modelin birden fazla metrik üzerinden karşılaştırma grafiği.

    Args:
        model_names: Model isimleri
        metrics_dict: {metrik_adı: [değerler]} — her liste model_names ile aynı uzunlukta
        output_path: Çıktı dosya yolu
        title: Grafik başlığı

    Returns:
        Kaydedilen dosya yolu
    """
    _apply_style()
    path = Path(output_path)

    n_models = len(model_names)
    n_metrics = len(metrics_dict)
    metric_names = list(metrics_dict.keys())

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, max(5, n_models * 0.6 + 2)))
    if n_metrics == 1:
        axes = [axes]

    colors = [MODEL_COLORS[i % len(MODEL_COLORS)] for i in range(n_models)]

    for ax, metric_name in zip(axes, metric_names):
        values = metrics_dict[metric_name]
        sorted_pairs = sorted(zip(model_names, values, colors), key=lambda x: x[1], reverse=True)

        s_names = [p[0] for p in sorted_pairs]
        s_values = [p[1] for p in sorted_pairs]
        s_colors = [p[2] for p in sorted_pairs]

        bars = ax.barh(range(n_models), s_values[::-1], color=s_colors[::-1],
                       edgecolor="white", linewidth=0.5, height=0.65)

        for bar, val in zip(bars, s_values[::-1]):
            ax.text(bar.get_width() + max(s_values) * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9, fontweight="bold")

        ax.set_yticks(range(n_models))
        ax.set_yticklabels(s_names[::-1], fontsize=10)

        # En iyi modele rozet
        best_idx = n_models - 1  # Çevrilmiş listede son = en iyi
        ax.get_yticklabels()[best_idx].set_fontweight("bold")
        ax.get_yticklabels()[best_idx].set_color(COLORS["primary"])

        ax.set_xlabel(metric_name.upper(), fontweight="bold")
        ax.set_title(metric_name.upper(), fontsize=13, fontweight="bold")
        ax.set_xlim(0, max(s_values) * 1.18 if max(s_values) > 0 else 1)

    fig.suptitle(f"📊 {title}", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save_figure(fig, path)


# ═══════════════════════════════════════════════════════════
#  7. SINIF DAĞILIMI
# ═══════════════════════════════════════════════════════════

def plot_class_distribution(
    y: np.ndarray,
    output_path: Union[str, Path] = "results/class_distribution.png",
    title: str = "Sınıf Dağılımı",
    labels: Optional[List[str]] = None,
) -> Path:
    """
    Hedef değişkenin sınıf dağılımı grafiği oluşturur.

    Args:
        y: Hedef değişken
        output_path: Çıktı dosya yolu
        title: Grafik başlığı
        labels: Sınıf isimleri

    Returns:
        Kaydedilen dosya yolu
    """
    _apply_style()
    path = Path(output_path)

    unique, counts = np.unique(y, return_counts=True)
    n_classes = len(unique)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    class_labels = labels if labels else [str(c) for c in unique]
    colors = [MODEL_COLORS[i % len(MODEL_COLORS)] for i in range(n_classes)]

    # Bar chart
    bars = ax1.bar(range(n_classes), counts, color=colors,
                   edgecolor="white", linewidth=1, width=0.6)
    for bar, count in zip(bars, counts):
        pct = count / len(y) * 100
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.02,
                 f"{count}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax1.set_xticks(range(n_classes))
    ax1.set_xticklabels(class_labels, fontsize=10)
    ax1.set_xlabel("Sınıf", fontweight="bold")
    ax1.set_ylabel("Örneklem Sayısı", fontweight="bold")
    ax1.set_title("Dağılım (Bar)", fontweight="bold")

    # Pie chart
    wedges, texts, autotexts = ax2.pie(
        counts, labels=class_labels, autopct="%1.1f%%",
        colors=colors, startangle=90, pctdistance=0.75,
        wedgeprops=dict(edgecolor="white", linewidth=2),
    )
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight("bold")

    # Donut efekti
    centre_circle = plt.Circle((0, 0), 0.50, fc="white")
    ax2.add_artist(centre_circle)
    ax2.text(0, 0, f"N={len(y)}", ha="center", va="center",
             fontsize=14, fontweight="bold", color=COLORS["neutral"])
    ax2.set_title("Dağılım (Pasta)", fontweight="bold")

    fig.suptitle(f"📊 {title} ({n_classes} sınıf)", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save_figure(fig, path)


# ═══════════════════════════════════════════════════════════
#  ANA SINIF — MLVisualizer
# ═══════════════════════════════════════════════════════════

class MLVisualizer:
    """
    Tüm ML görselleştirmelerini tek bir sınıfta toplayan yardımcı.

    Kullanım:
        viz = MLVisualizer(output_dir="results/plots")
        saved = viz.plot_all(
            model, X_train, X_test, y_train, y_test,
            feature_names=feature_cols,
            df=df
        )
    """

    def __init__(self, output_dir: str = "results/plots", prefix: str = ""):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = f"{prefix}_" if prefix else ""
        self.saved_plots: Dict[str, Path] = {}

    def _path(self, name: str) -> Path:
        return self.output_dir / f"{self.prefix}{name}.png"

    def plot_all(
        self,
        model: BaseEstimator,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[List[str]] = None,
        df: Optional[pd.DataFrame] = None,
        task_type: str = "classification",
        model_name: str = "Model",
    ) -> Dict[str, Path]:
        """
        Tüm grafikleri tek çağrıda oluşturur.

        Args:
            model: Eğitilmiş model
            X_train, X_test, y_train, y_test: Veri
            feature_names: Özellik isimleri
            df: Orijinal DataFrame (korelasyon matrisi için)
            task_type: "classification" veya "regression"
            model_name: Model adı (başlıklarda kullanılır)

        Returns:
            {grafik_adı: dosya_yolu} dict'i
        """
        print(f"\n🎨 Görselleştirmeler oluşturuluyor ({model_name})...")
        print("─" * 50)

        y_pred = model.predict(X_test)
        X_all = np.vstack([X_train, X_test])
        y_all = np.concatenate([y_train, y_test])

        # 1. Sınıf dağılımı
        self.saved_plots["class_distribution"] = plot_class_distribution(
            y_all, self._path("class_distribution"),
            title=f"Sınıf Dağılımı — {model_name}",
        )

        if task_type == "classification":
            # 2. Confusion Matrix
            self.saved_plots["confusion_matrix"] = plot_confusion_matrix(
                y_test, y_pred, self._path("confusion_matrix"),
                title=f"Confusion Matrix — {model_name}",
            )

            # 2b. Normalize confusion matrix
            self.saved_plots["confusion_matrix_norm"] = plot_confusion_matrix(
                y_test, y_pred, self._path("confusion_matrix_normalized"),
                title=f"Confusion Matrix — {model_name}",
                normalize=True,
            )

            # 3. ROC Curve
            if hasattr(model, "predict_proba"):
                self.saved_plots["roc_curve"] = plot_roc_curve(
                    model, X_test, y_test, self._path("roc_curve"),
                    title=f"ROC Eğrisi — {model_name}",
                )

        # 4. Feature Importance
        if feature_names:
            self.saved_plots["feature_importance"] = plot_feature_importance(
                model, feature_names, self._path("feature_importance"),
                title=f"Özellik Önemliliği — {model_name}",
            )

        # 5. Korelasyon Matrisi
        if df is not None:
            self.saved_plots["correlation_matrix"] = plot_correlation_matrix(
                df, self._path("correlation_matrix"),
                title="Korelasyon Matrisi",
            )

        # 6. Learning Curve
        scoring = "accuracy" if task_type == "classification" else "r2"
        self.saved_plots["learning_curve"] = plot_learning_curve(
            model, X_all, y_all, self._path("learning_curve"),
            title=f"Öğrenme Eğrisi — {model_name}",
            scoring=scoring,
        )

        print("─" * 50)
        print(f"✅ Toplam {len(self.saved_plots)} grafik oluşturuldu → {self.output_dir}/")

        return self.saved_plots

    def get_summary(self) -> str:
        """Oluşturulan grafiklerin özetini döndürür."""
        lines = ["## 📊 Oluşturulan Grafikler\n"]
        for name, path in self.saved_plots.items():
            display_name = name.replace("_", " ").title()
            lines.append(f"- **{display_name}:** `{path}`")
        return "\n".join(lines)
