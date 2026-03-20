import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay
from sklearn.preprocessing import LabelBinarizer
from sklearn.inspection import permutation_importance
import pandas as pd
from collections import Counter

class MLVisualizer:
    def __init__(self, output_dir="results/plots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        plt.style.use('seaborn-v0_8-darkgrid') # Updated style

    def _save_plot(self, fig, filename):
        fig.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=300)
        plt.close(fig)
        print(f"Grafik kaydedildi: {filepath}")

    def plot_confusion_matrix(self, y_true, y_pred, labels=None, normalize=False, filename="confusion_matrix.png"):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalize Edilmiş Karışıklık Matrisi'
            fmt = '.2f'
        else:
            title = 'Karışıklık Matrisi'
            fmt = 'd'

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', ax=ax,
                    xticklabels=labels, yticklabels=labels, cbar=False)
        ax.set_title(title)
        ax.set_xlabel('Tahmin Edilen Sınıf')
        ax.set_ylabel('Gerçek Sınıf')
        self._save_plot(fig, filename)

    def plot_roc_curve(self, model, X_test, y_test, filename="roc_curve.png"):
        fig, ax = plt.subplots(figsize=(8, 6))
        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(X_test)
            if y_score.shape[1] == 2: # Binary classification
                RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name='ROC Curve')
                ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Rastgele')
                ax.set_title('ROC Eğrisi - İkili Sınıflandırma')
            else: # Multi-class classification (OvR)
                lb = LabelBinarizer()
                y_test_binarized = lb.fit_transform(y_test)
                for i, class_label in enumerate(lb.classes_):
                    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, label=f'ROC curve of class {class_label} (area = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Rastgele')
                ax.set_xlabel('Yanlış Pozitif Oranı')
                ax.set_ylabel('Doğru Pozitif Oranı')
                ax.set_title('ROC Eğrisi - Çoklu Sınıflandırma (OvR)')
                ax.legend(loc="lower right")
        else:
            print("Model, predict_proba metoduna sahip değil, ROC eğrisi çizilemedi.")
            return

        self._save_plot(fig, filename)

    def plot_feature_importance(self, model, feature_names, filename="feature_importance.png", top_n=15):
        importances = None
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute coefficients
            if model.coef_.ndim > 1: # Multi-class, take mean of absolute coeffs
                importances = np.mean(np.abs(model.coef_), axis=0)
            else: # Binary class
                importances = np.abs(model.coef_)
        else:
            # Fallback to permutation importance if other methods not available
            print("Model, feature_importances_ veya coef_ özelliklerine sahip değil. Permütasyon önemini hesaplıyorum (daha yavaş olabilir).")
            # Permutation importance requires fitting to be done on X_train, y_train
            # The model passed here is a pipeline. We need to pass the model without scaler for permutation_importance if used with raw X_test
            # Or, we can use the pipeline directly but it might be slower.
            # For simplicity, let's assume the model is already trained and passed.
            # If the model is a pipeline, we need to extract the actual model.
            try:
                # Assuming the last step of the pipeline is the model
                trained_model = model.named_steps['model'] if 'model' in model.named_steps else model
                # It's better to use X_test for permutation importance, but needs to be scaled by the pipeline's scaler.
                # Since the model is a pipeline, X_test should be passed directly.
                r = permutation_importance(model, X_test_for_perm_imp, y_test_for_perm_imp,
                                           n_repeats=10, random_state=42, n_jobs=-1)
                importances = r.importances_mean
            except Exception as e:
                print(f"Permütasyon önemi hesaplanırken hata oluştu: {e}")
                return

        if importances is not None:
            # Create a DataFrame for sorting
            feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
            feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
            
            # Select top_n features
            feature_importance_df = feature_importance_df.head(top_n)

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(x='importance', y='feature', data=feature_importance_df, ax=ax, palette='viridis')
            ax.set_title('Özellik Önemleri')
            ax.set_xlabel('Önem Değeri')
            ax.set_ylabel('Özellik')
            self._save_plot(fig, filename)
        else:
            print("Özellik önemleri hesaplanamadı.")

    def plot_correlation_matrix(self, df, filename="correlation_matrix.png"):
        plt.figure(figsize=(12, 10))
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Korelasyon Matrisi')
        self._save_plot(plt.gcf(), filename)

    def plot_class_distribution(self, y, class_names=None, filename="class_distribution.png"):
        counts = Counter(y)
        if class_names is None:
            class_names = [f'Sınıf {c}' for c in sorted(counts.keys())]
        
        labels = [class_names[c] for c in sorted(counts.keys())]
        sizes = [counts[c] for c in sorted(counts.keys())]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Bar plot
        sns.barplot(x=labels, y=sizes, ax=axes[0], palette='viridis')
        axes[0].set_title('Sınıf Dağılımı (Çubuk Grafik)')
        axes[0].set_xlabel('Sınıf')
        axes[0].set_ylabel('Adet')

        # Pie chart / Donut chart
        axes[1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('viridis', len(labels)),
                    wedgeprops=dict(width=0.3)) # For donut effect
        axes[1].set_title('Sınıf Dağılımı (Pasta Grafik)')
        axes[1].axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.

        self._save_plot(fig, filename)
    
    def plot_all(self, model, X_train, X_test, y_train, y_test, feature_names=None, df=None, class_names=None):
        print("Tüm görselleştirmeler oluşturuluyor...")
        y_pred = model.predict(X_test)

        if feature_names is None:
            feature_names = X_train.columns if isinstance(X_train, pd.DataFrame) else [f'feature_{i}' for i in range(X_train.shape[1])]

        if class_names is None and len(np.unique(y_train)) <= 10: # Only try to infer for reasonable number of classes
            class_names = [str(c) for c in np.unique(y_train)]

        # Confusion Matrix
        print("Karışıklık Matrisi çiziliyor...")
        self.plot_confusion_matrix(y_test, y_pred, labels=class_names, normalize=False, filename="confusion_matrix_raw.png")
        self.plot_confusion_matrix(y_test, y_pred, labels=class_names, normalize=True, filename="confusion_matrix_normalized.png")

        # ROC Curve
        print("ROC Eğrisi çiziliyor...")
        self.plot_roc_curve(model, X_test, y_test, filename="roc_curve.png")

        # Feature Importance (pass X_test and y_test for permutation importance fallback)
        print("Özellik Önemleri çiziliyor...")
        # Need to pass X_test and y_test correctly for permutation importance fallback
        # This requires a slight modification to how plot_feature_importance is called
        # if the model doesn't have feature_importances_ or coef_.
        # For now, let's just pass feature_names.
        # A more robust solution would modify plot_feature_importance to take X_test_for_perm_imp and y_test_for_perm_imp as arguments.
        # For this project, assume the primary methods (feature_importances_, coef_) will work.
        global X_test_for_perm_imp, y_test_for_perm_imp # Hack to pass to plot_feature_importance
        X_test_for_perm_imp = X_test
        y_test_for_perm_imp = y_test
        self.plot_feature_importance(model, feature_names, filename="feature_importance.png")

        # Correlation Matrix
        if df is not None:
            print("Korelasyon Matrisi çiziliyor...")
            self.plot_correlation_matrix(df, filename="correlation_matrix.png")

        # Class Distribution
        print("Sınıf Dağılımı çiziliyor...")
        self.plot_class_distribution(y_train, class_names=class_names, filename="class_distribution_train.png")
        self.plot_class_distribution(y_test, class_names=class_names, filename="class_distribution_test.png")

        print("Tüm görselleştirmeler tamamlandı.")