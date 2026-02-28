# utils/visualize.py
# ═══════════════════════════════════════════════════════════
#  Bio-ML Agent — Etkileşimli Görselleştirme Modülü (Plotly)
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
# ═══════════════════════════════════════════════════════════

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.base import BaseEstimator
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  Ortak Stil Yapılandırması
# ─────────────────────────────────────────────

COLORS = {
    "primary": "#6366F1",       # Indigo
    "secondary": "#8B5CF6",     # Violet
    "success": "#10B981",       # Emerald
    "warning": "#F59E0B",       # Amber
    "danger": "#EF4444",        # Red
    "info": "#06B6D4",          # Cyan
    "neutral": "#64748B",       # Slate
}

MODEL_COLORS = [
    "#6366F1", "#10B981", "#F59E0B", "#EF4444",
    "#8B5CF6", "#06B6D4", "#EC4899", "#14B8A6",
    "#F97316", "#3B82F6", "#84CC16", "#A855F7",
]

def _save_figure(fig: go.Figure, path: Path) -> Path:
    """Plotly figürünü interaktif HTML olarak kaydeder."""
    path = path.with_suffix('.html')
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Clean transparent background and beautiful default layout
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, sans-serif", color="#334155"),
        title_font=dict(size=20, color="#1E293B"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,0.5)",
        margin=dict(t=80, l=50, r=50, b=50),
        hoverlabel=dict(bgcolor="white", font_size=13, font_family="Inter")
    )
    
    fig.write_html(str(path), include_plotlyjs='cdn', full_html=False)
    print(f"  📊 İnteraktif Grafik kaydedildi: {path}")
    return path


# ═══════════════════════════════════════════════════════════
#  1. CONFUSION MATRIX
# ═══════════════════════════════════════════════════════════

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Union[str, Path] = "results/confusion_matrix.html",
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    normalize: bool = False,
) -> Path:
    path = Path(output_path)
    cm = confusion_matrix(y_true, y_pred)
    tick_labels = labels if labels else [str(c) for c in sorted(np.unique(np.concatenate([y_true, y_pred])))]
    
    if normalize:
        cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
        display_cm = np.nan_to_num(cm_norm)
        title_suffix = " (Normalize)"
        text_template = "%{z:.2f}"
    else:
        display_cm = cm
        title_suffix = ""
        text_template = "%{z:d}"

    fig = go.Figure(data=go.Heatmap(
        z=display_cm[::-1], # Plotly heatmaps draw from bottom-up
        x=tick_labels,
        y=tick_labels[::-1],
        hoverongaps=False,
        colorscale="Blues",
        text=display_cm[::-1],
        texttemplate=text_template,
        textfont={"size": 14}
    ))
    
    fig.update_layout(
        title=f"🔲 {title}{title_suffix}",
        xaxis_title="Tahmin Edilen Sınıf",
        yaxis_title="Gerçek Sınıf",
        xaxis=dict(side="bottom")
    )
    return _save_figure(fig, path)


# ═══════════════════════════════════════════════════════════
#  2. ROC CURVE
# ═══════════════════════════════════════════════════════════

def plot_roc_curve(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_path: Union[str, Path] = "results/roc_curve.html",
    title: str = "ROC Eğrisi",
    labels: Optional[List[str]] = None,
) -> Path:
    path = Path(output_path)
    if not hasattr(model, "predict_proba"):
        print(f"  ⚠️ Model predict_proba desteklemiyor, ROC eğrisi oluşturulamadı.")
        return path

    classes = np.unique(y_test)
    n_classes = len(classes)
    fig = go.Figure()

    if n_classes == 2:
        y_score = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=classes[1])
        roc_auc = auc(fpr, tpr)
        label_name = labels[1] if labels and len(labels) > 1 else f"Sınıf {classes[1]}"
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, name=f"{label_name} (AUC = {roc_auc:.4f})",
            mode='lines', line=dict(color=COLORS['primary'], width=3),
            fill='tozeroy', fillcolor=f"rgba(99, 102, 241, 0.15)"
        ))
    else:
        y_bin = label_binarize(y_test, classes=classes)
        y_score = model.predict_proba(X_test)
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            color = MODEL_COLORS[i % len(MODEL_COLORS)]
            label_name = labels[i] if labels and i < len(labels) else f"Sınıf {cls}"
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, name=f"{label_name} (AUC = {roc_auc:.3f})",
                mode='lines', line=dict(color=color, width=2)
            ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], name="Rastgele Tahmin",
        mode='lines', line=dict(color='black', width=2, dash='dash')
    ))

    fig.update_layout(
        title=f"📈 {title}",
        xaxis_title="Yanlış Pozitif Oranı (FPR)",
        yaxis_title="Doğru Pozitif Oranı (TPR)",
        xaxis=dict(range=[-0.02, 1.02]),
        yaxis=dict(range=[-0.02, 1.02]),
        legend=dict(x=0.65, y=0.05, bgcolor="rgba(255,255,255,0.8)")
    )
    return _save_figure(fig, path)


# ═══════════════════════════════════════════════════════════
#  3. FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════

def plot_feature_importance(
    model: BaseEstimator,
    feature_names: List[str],
    output_path: Union[str, Path] = "results/feature_importance.html",
    title: str = "Özellik Önemliliği",
    top_n: int = 20,
) -> Path:
    path = Path(output_path)
    estimator = model
    if hasattr(model, "named_steps"):
        step_names = list(model.named_steps.keys())
        estimator = model.named_steps[step_names[-1]]
    elif hasattr(model, "steps"):
        estimator = model.steps[-1][1]

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
        imp_type = "Gini Importance"
    elif hasattr(estimator, "coef_"):
        coef = estimator.coef_
        importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef.ravel())
        imp_type = "|Katsayı| (Ortalama)"
    else:
        fig = go.Figure()
        fig.add_annotation(text="Bu model özellik önemliliği desteklemiyor", showarrow=False, font=dict(size=20))
        return _save_figure(fig, path)

    n_features = min(top_n, len(feature_names), len(importances))
    indices = np.argsort(importances)[::-1][:n_features]
    
    sorted_names = [feature_names[i] for i in indices][::-1]
    sorted_values = importances[indices][::-1]

    fig = go.Figure(go.Bar(
        x=sorted_values, y=sorted_names,
        orientation='h',
        marker=dict(
            color=sorted_values,
            colorscale='Viridis',
            line=dict(color='white', width=1)
        ),
        text=[f"{v:.4f}" for v in sorted_values],
        textposition='outside'
    ))

    fig.update_layout(
        title=f"🎯 {title} (Top {n_features})",
        xaxis_title=imp_type,
        yaxis_title="Özellik",
        height=max(500, n_features * 30 + 100)
    )
    return _save_figure(fig, path)


# ═══════════════════════════════════════════════════════════
#  4. KORELASYON MATRİSİ
# ═══════════════════════════════════════════════════════════

def plot_correlation_matrix(
    df: pd.DataFrame,
    output_path: Union[str, Path] = "results/correlation_matrix.html",
    title: str = "Korelasyon Matrisi",
    method: str = "pearson",
    annot_threshold: int = 15,
) -> Path:
    path = Path(output_path)
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr(method=method)

    fig = go.Figure(data=go.Heatmap(
        z=corr.values[::-1],
        x=corr.columns,
        y=corr.columns[::-1],
        colorscale='RdBu',
        zmin=-1, zmax=1,
        text=np.round(corr.values[::-1], 2),
        texttemplate="%{text}" if len(corr.columns) <= annot_threshold else None,
        hoverinfo="x+y+z"
    ))

    fig.update_layout(
        title=f"🔥 {title} ({method.capitalize()})",
        xaxis_title="Özellikler",
        yaxis_title="Özellikler",
        height=max(600, len(corr.columns)*25 + 200),
        width=max(600, len(corr.columns)*25 + 200),
        yaxis=dict(tickangle=0),
        xaxis=dict(tickangle=-45)
    )
    return _save_figure(fig, path)


# ═══════════════════════════════════════════════════════════
#  5. LEARNING CURVE
# ═══════════════════════════════════════════════════════════

def plot_learning_curve(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    output_path: Union[str, Path] = "results/learning_curve.html",
    title: str = "Öğrenme Eğrisi",
    cv: int = 5,
    scoring: str = "accuracy",
    n_points: int = 10,
) -> Path:
    path = Path(output_path)
    train_sizes = np.linspace(0.1, 1.0, n_points)

    try:
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y, train_sizes=train_sizes, cv=cv, scoring=scoring, n_jobs=1
        )
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Learning curve hesaplanamadı:<br>{e}", showarrow=False, font=dict(color="red"))
        return _save_figure(fig, path)

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig = go.Figure()

    # Eğitim eğrisi
    fig.add_trace(go.Scatter(
        x=train_sizes_abs, y=train_mean,
        name=f"Eğitim {scoring}",
        mode='lines+markers',
        line=dict(color=COLORS["primary"], width=3)
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes_abs, train_sizes_abs[::-1]]),
        y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
        fill='toself', fillcolor='rgba(99, 102, 241, 0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False, hoverinfo='skip'
    ))

    # Doğrulama eğrisi
    fig.add_trace(go.Scatter(
        x=train_sizes_abs, y=val_mean,
        name=f"Doğrulama {scoring}",
        mode='lines+markers',
        line=dict(color=COLORS["success"], width=3)
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes_abs, train_sizes_abs[::-1]]),
        y=np.concatenate([val_mean + val_std, (val_mean - val_std)[::-1]]),
        fill='toself', fillcolor='rgba(16, 185, 129, 0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False, hoverinfo='skip'
    ))

    # Overfit tespiti rozeti
    gap = train_mean[-1] - val_mean[-1]
    status = "⚠️ Overfitting" if gap > 0.1 else "⚠️ Underfitting" if val_mean[-1] < 0.6 else "✅ İyi Genelleme"
    status_color = "darkred" if "⚠️" in status else "darkgreen"
    
    fig.add_annotation(
        x=1, y=0, xref="paper", yref="paper",
        text=f"Boşluk: {gap:.4f}<br>{status}",
        showarrow=False, bgcolor="white", bordercolor=status_color,
        borderwidth=2, font=dict(color=status_color, size=14),
        xanchor="right", yanchor="bottom", borderpad=8
    )

    fig.update_layout(
         title=f"📉 {title}", xaxis_title="Eğitim Örnekleri Sayısı", yaxis_title=scoring.capitalize()
    )
    return _save_figure(fig, path)


# ═══════════════════════════════════════════════════════════
#  6. MODEL KARŞILAŞTIRMA (Bar Chart)
# ═══════════════════════════════════════════════════════════

def plot_model_comparison(
    model_names: List[str],
    metrics_dict: Dict[str, List[float]],
    output_path: Union[str, Path] = "results/model_comparison.html",
    title: str = "Model Karşılaştırması",
) -> Path:
    path = Path(output_path)
    metric_names = list(metrics_dict.keys())
    
    fig = make_subplots(rows=1, cols=len(metric_names), subplot_titles=[m.upper() for m in metric_names])
    
    for i, metric in enumerate(metric_names):
        values = metrics_dict[metric]
        sorted_pairs = sorted(zip(model_names, values), key=lambda x: x[1])
        s_names = [p[0] for p in sorted_pairs]
        s_vals = [p[1] for p in sorted_pairs]
        
        colors = [COLORS["primary"] if j == len(s_names)-1 else COLORS["neutral"] for j in range(len(s_names))]
        
        fig.add_trace(
            go.Bar(x=s_vals, y=s_names, orientation='h', marker_color=colors, 
                   text=[f"{v:.4f}" for v in s_vals], textposition='auto'),
            row=1, col=i+1
        )

    fig.update_layout(
        title=f"📊 {title}",
        showlegend=False,
        height=max(500, len(model_names)*45 + 100)
    )
    return _save_figure(fig, path)


# ═══════════════════════════════════════════════════════════
#  7. SINIF DAĞILIMI
# ═══════════════════════════════════════════════════════════

def plot_class_distribution(
    y: np.ndarray,
    output_path: Union[str, Path] = "results/class_distribution.html",
    title: str = "Sınıf Dağılımı",
    labels: Optional[List[str]] = None,
) -> Path:
    path = Path(output_path)
    unique, counts = np.unique(y, return_counts=True)
    class_labels = labels if labels else [str(c) for c in unique]
    colors = [MODEL_COLORS[i % len(MODEL_COLORS)] for i in range(len(unique))]

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "domain"}]],
                        subplot_titles=("Dağılım (Bar)", "Dağılım (Pasta)"))

    # Bar
    fig.add_trace(go.Bar(x=class_labels, y=counts, marker_color=colors,
                         text=counts, textposition='auto', name="Adet"), row=1, col=1)
    
    # Donut Pie
    fig.add_trace(go.Pie(labels=class_labels, values=counts, hole=.4, 
                         marker=dict(colors=colors, line=dict(color='white', width=2)),
                         textinfo='label+percent', name="Oran"), row=1, col=2)

    fig.add_annotation(x=0.8, y=0.5, text=f"N={len(y)}", showarrow=False, 
                       font=dict(size=20, weight="bold"), xref="paper", yref="paper")

    fig.update_layout(title=f"📊 {title}")
    return _save_figure(fig, path)


# ═══════════════════════════════════════════════════════════
#  ANA SINIF — MLVisualizer
# ═══════════════════════════════════════════════════════════

class MLVisualizer:
    def __init__(self, output_dir: str = "results/plots", prefix: str = ""):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = f"{prefix}_" if prefix else ""
        self.saved_plots: Dict[str, Path] = {}

    def _path(self, name: str) -> Path:
        return self.output_dir / f"{self.prefix}{name}.html"

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

        print(f"\n🎨 İnteraktif Görseller oluşturuluyor ({model_name})...")
        print("─" * 50)

        y_pred = model.predict(X_test)
        X_all = np.vstack([X_train, X_test])
        y_all = np.concatenate([y_train, y_test])

        self.saved_plots["class_distribution"] = plot_class_distribution(
            y_all, self._path("class_distribution"), title=f"Sınıf Dağılımı — {model_name}"
        )

        if task_type == "classification":
            self.saved_plots["confusion_matrix"] = plot_confusion_matrix(
                y_test, y_pred, self._path("confusion_matrix"), title=f"Confusion Matrix — {model_name}"
            )
            self.saved_plots["confusion_matrix_norm"] = plot_confusion_matrix(
                y_test, y_pred, self._path("confusion_matrix_normalized"), 
                title=f"Confusion Matrix — {model_name}", normalize=True
            )
            if hasattr(model, "predict_proba"):
                self.saved_plots["roc_curve"] = plot_roc_curve(
                    model, X_test, y_test, self._path("roc_curve"), title=f"ROC Eğrisi — {model_name}"
                )

        if feature_names:
            self.saved_plots["feature_importance"] = plot_feature_importance(
                model, feature_names, self._path("feature_importance"), title=f"Özellik Önemliliği — {model_name}"
            )

        if df is not None:
            self.saved_plots["correlation_matrix"] = plot_correlation_matrix(
                df, self._path("correlation_matrix"), title="Korelasyon Matrisi"
            )

        scoring = "accuracy" if task_type == "classification" else "r2"
        self.saved_plots["learning_curve"] = plot_learning_curve(
            model, X_all, y_all, self._path("learning_curve"), title=f"Öğrenme Eğrisi — {model_name}", scoring=scoring
        )

        print("─" * 50)
        print(f"✅ Toplam {len(self.saved_plots)} interaktif grafik oluşturuldu → {self.output_dir}/")
        return self.saved_plots

    def get_summary(self) -> str:
        lines = ["## 📊 Oluşturulan İnteraktif Grafikler (HTML)\n"]
        for name, path in self.saved_plots.items():
            display_name = name.replace("_", " ").title()
            lines.append(f"- **{display_name}:** `{path}`")
        return "\n".join(lines)
