import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
from sklearn.model_selection import learning_curve
import os

class MLVisualizer:
    def __init__(self, output_dir="results/plots"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _save_plot(self, fig, filename):
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, filename))
        plt.close(fig)

    def plot_confusion_matrix(self, model, X_test, y_test, title="Karışıklık Matrisi", filename="confusion_matrix.png"):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Normal Confusion Matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot(cmap=plt.cm.Blues, ax=axes[0])
        axes[0].set_title(f'{title} (Normal)')

        # Normalized Confusion Matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        disp_normalized = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=model.classes_)
        disp_normalized.plot(cmap=plt.cm.Blues, ax=axes[1])
        axes[1].set_title(f'{title} (Normalize Edilmiş)')

        self._save_plot(fig, filename)
        print(f"'{filename}' kaydedildi.")

    def plot_roc_curve(self, model, X_test, y_test, title="ROC Eğrisi", filename="roc_curve.png"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Yanlış Pozitif Oranı')
        ax.set_ylabel('Doğru Pozitif Oranı')
        ax.set_title(title)
        ax.legend(loc="lower right")
        self._save_plot(fig, filename)
        print(f"'{filename}' kaydedildi.")

    def plot_feature_importance(self, model, feature_names, title="Özellik Önemleri", filename="feature_importance.png"):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models like Logistic Regression or SVC with linear kernel
            # Assuming binary classification, taking the absolute value of the first coef_ array
            importances = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
        else:
            print("Modelin özellik önemlerini veya katsayılarını çıkaramıyor.")
            return

        feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance_df, ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Önem Derecesi')
        ax.set_ylabel('Özellik')
        self._save_plot(fig, filename)
        print(f"'{filename}' kaydedildi.")

    def plot_correlation_matrix(self, df, title="Korelasyon Matrisi", filename="correlation_matrix.png"):
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
        ax.set_title(title)
        self._save_plot(fig, filename)
        print(f"'{filename}' kaydedildi.")

    def plot_learning_curve(self, model, X, y, title="Öğrenme Eğrisi", filename="learning_curve.png"):
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), random_state=42,
            scoring='accuracy' # Use appropriate scoring
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
        ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Eğitim Skoru")
        ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Çapraz Doğrulama Skoru")
        ax.set_xlabel("Eğitim Örnek Sayısı")
        ax.set_ylabel("Skor")
        ax.set_title(title)
        ax.legend(loc="best")
        self._save_plot(fig, filename)
        print(f"'{filename}' kaydedildi.")

    def plot_class_distribution(self, y, class_names=None, title="Sınıf Dağılımı", filename="class_distribution.png"):
        if class_names is None:
            class_names = [str(c) for c in np.unique(y)]
        
        counts = pd.Series(y).value_counts().sort_index()
        labels = [class_names[c] for c in counts.index]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Bar plot
        sns.barplot(x=labels, y=counts.values, ax=axes[0])
        axes[0].set_title(f'{title} (Çubuk Grafik)')
        axes[0].set_xlabel('Sınıf')
        axes[0].set_ylabel('Sayı')

        # Donut plot
        axes[1].pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.3))
        axes[1].set_title(f'{title} (Donut Grafik)')
        axes[1].axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.

        self._save_plot(fig, filename)
        print(f"'{filename}' kaydedildi.")

    def plot_all(self, best_model_pipeline, X_train, X_test, y_train, y_test, feature_names, df, class_names=None):
        print("\n--- Görselleştirmeler Oluşturuluyor ---")
        self.plot_confusion_matrix(best_model_pipeline, X_test, y_test, title="En İyi Model Karışıklık Matrisi")
        self.plot_roc_curve(best_model_pipeline, X_test, y_test, title="En İyi Model ROC Eğrisi")
        self.plot_feature_importance(best_model_pipeline.named_steps['model'], feature_names, title="En İyi Model Özellik Önemleri")
        self.plot_correlation_matrix(df, title="Veri Seti Korelasyon Matrisi")
        self.plot_learning_curve(best_model_pipeline, pd.concat([X_train, X_test]), pd.concat([y_train, y_test]), title="En İyi Model Öğrenme Eğrisi")
        self.plot_class_distribution(y_train, class_names=class_names, title="Eğitim Verisi Sınıf Dağılımı")