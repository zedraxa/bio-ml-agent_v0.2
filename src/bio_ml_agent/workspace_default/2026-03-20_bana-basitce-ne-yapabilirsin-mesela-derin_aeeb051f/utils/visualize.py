import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import learning_curve
import joblib

class MLVisualizer:
    """
    Makine öğrenmesi modelleri ve veri setleri için görselleştirmeler oluşturan bir yardımcı sınıf.
    """
    def __init__(self, output_dir="results/plots"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        print(f"Grafikler '{self.output_dir}' klasörüne kaydedilecek.")

    def _save_plot(self, fig, filename):
        """Yardımcı fonksiyon: Grafiği kaydeder."""
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f"  - Grafik kaydedildi: {filename}")

    def plot_correlation_heatmap(self, df, title='Özellik Korelasyon Matrisi', filename='correlation_heatmap.png'):
        """Veri setindeki özelliklerin korelasyon matrisini çizer."""
        print("Korelasyon matrisi oluşturuluyor...")
        fig, ax = plt.subplots(figsize=(18, 15))
        corr = df.corr()
        sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax)
        ax.set_title(title, fontsize=16)
        self._save_plot(fig, filename)

    def plot_class_distribution(self, y, target_map={0: 'Benign', 1: 'Malignant'}, filename_prefix='class_distribution'):
        """Hedef değişkenin sınıf dağılımını bar ve donut grafiği olarak çizer."""
        print("Sınıf dağılım grafikleri oluşturuluyor...")
        target_counts = y.value_counts()
        labels = target_counts.index.map(target_map)
        
        # Bar Plot
        fig_bar, ax_bar = plt.subplots(figsize=(8, 6))
        sns.barplot(x=labels, y=target_counts.values, ax=ax_bar)
        ax_bar.set_title('Sınıf Dağılımı (Bar Grafiği)', fontsize=14)
        ax_bar.set_ylabel('Sayı')
        self._save_plot(fig_bar, f'{filename_prefix}_bar.png')

        # Donut Plot
        fig_donut, ax_donut = plt.subplots(figsize=(8, 8))
        ax_donut.pie(target_counts, labels=labels, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.3))
        ax_donut.set_title('Sınıf Dağılımı (Donut Grafiği)', fontsize=14)
        ax_donut.axis('equal')
        self._save_plot(fig_donut, f'{filename_prefix}_donut.png')

    def plot_confusion_matrix(self, y_true, y_pred, class_names, normalize=False, filename='confusion_matrix.png'):
        """Karışıklık matrisini çizer."""
        print(f"Karışıklık matrisi oluşturuluyor (Normalize={normalize})...")
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalize Edilmiş Karışıklık Matrisi'
        else:
            fmt = 'd'
            title = 'Karışıklık Matrisi'

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_ylabel('Gerçek Etiket')
        ax.set_xlabel('Tahmin Edilen Etiket')
        ax.set_title(title, fontsize=14)
        self._save_plot(fig, filename)

    def plot_roc_curve(self, y_true, y_pred_proba, model_name='Model', filename='roc_curve.png'):
        """ROC eğrisini çizer."""
        print("ROC eğrisi oluşturuluyor...")
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Eğrisi (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{model_name} için ROC Eğrisi', fontsize=14)
        ax.legend(loc="lower right")
        self._save_plot(fig, filename)

    def plot_feature_importance(self, model, feature_names, top_n=15, filename='feature_importance.png'):
        """Modelin özellik önem sıralamasını çizer."""
        print("Özellik önem sıralaması grafiği oluşturuluyor...")
        if not hasattr(model, 'coef_') and not hasattr(model, 'feature_importances_'):
            print(f"  - Uyarı: Model ({type(model).__name__}) 'coef_' veya 'feature_importances_' özniteliğine sahip değil. Grafik atlanıyor.")
            return

        if hasattr(model, 'feature_importances_'): # RandomForest, GradientBoosting
            importances = model.feature_importances_
        else: # LogisticRegression, SVM (linear)
            importances = np.abs(model.coef_[0])
        
        indices = np.argsort(importances)[::-1]
        
        df_importance = pd.DataFrame({
            'feature': [feature_names[i] for i in indices],
            'importance': [importances[i] for i in indices]
        }).head(top_n)

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=df_importance, ax=ax)
        ax.set_title('En Önemli Özellikler', fontsize=14)
        self._save_plot(fig, filename)

    def plot_learning_curve(self, estimator, X, y, cv=5, n_jobs=-1, filename='learning_curve.png'):
        """Öğrenme eğrisini çizer."""
        print("Öğrenme eğrisi oluşturuluyor...")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=np.linspace(.1, 1.0, 5))
        
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1, color="r")
        ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="g")
        ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Eğitim Skoru")
        ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Doğrulama Skoru")
        ax.set_title('Öğrenme Eğrisi', fontsize=14)
        ax.set_xlabel('Eğitim Örnekleri')
        ax.set_ylabel('Skor')
        ax.legend(loc="best")
        ax.grid()
        self._save_plot(fig, filename)

    def plot_all(self, model, X_train, X_test, y_train, y_test, feature_names, df, class_names=['Benign', 'Malignant']):
        """Tüm görselleştirmeleri sırayla oluşturur."""
        print("\n--- Tüm Görselleştirmeler Oluşturuluyor ---")
        
        # Veri setine dayalı grafikler
        self.plot_correlation_heatmap(df.drop('id', axis=1))
        self.plot_class_distribution(df['diagnosis'], target_map={0: 'Benign', 1: 'Malignant'})
        
        # Model performansına dayalı grafikler
        pipeline = joblib.load(model) if isinstance(model, str) else model
        
        # Pipeline'dan gerçek modeli ve ölçekleyiciyi çıkar
        scaler = pipeline.named_steps['scaler']
        final_model = pipeline.named_steps['model']
        
        X_test_scaled = scaler.transform(X_test)
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

        self.plot_confusion_matrix(y_test, y_pred, class_names=class_names, normalize=False, filename='confusion_matrix.png')
        self.plot_confusion_matrix(y_test, y_pred, class_names=class_names, normalize=True, filename='confusion_matrix_normalized.png')
        self.plot_roc_curve(y_test, y_pred_proba, model_name=type(final_model).__name__)
        self.plot_feature_importance(final_model, feature_names)
        self.plot_learning_curve(pipeline, X_train, y_train)

        print("--- Görselleştirme Tamamlandı ---")
