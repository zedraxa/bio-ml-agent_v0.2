import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
import pandas as pd
import os

class MLVisualizer:
    def __init__(self, output_dir="results/plots"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_confusion_matrix(self, y_true, y_pred, model_name="Model", normalize=False):
        cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues", cbar=False,
                    xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
        title = f"{model_name} Confusion Matrix"
        if normalize:
            title += " (Normalized)"
        plt.title(title)
        plt.xlabel("Tahmin Edilen")
        plt.ylabel("Gerçek")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{model_name.replace(' ', '_').lower()}_confusion_matrix{'_normalized' if normalize else ''}.png"))
        plt.close()

    def plot_roc_curve(self, y_true, y_proba, model_name="Model"):
        if len(np.unique(y_true)) < 2:
            print(f"Uyarı: ROC eğrisi için en az iki sınıf gereklidir. Model '{model_name}' için çizilemedi.")
            return
        
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC eğrisi (alan = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Yanlış Pozitif Oranı')
        plt.ylabel('Doğru Pozitif Oranı')
        plt.title(f'{model_name} ROC Eğrisi')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{model_name.replace(' ', '_').lower()}_roc_curve.png"))
        plt.close()

    def plot_feature_importance(self, model_pipeline, feature_names, model_name="Model"):
        model = model_pipeline.named_steps['model']
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(10, 6))
            sns.barplot(x=importances[indices], y=np.array(feature_names)[indices])
            plt.title(f"{model_name} Feature Importance")
            plt.xlabel("Önem Derecesi")
            plt.ylabel("Özellik")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"{model_name.replace(' ', '_').lower()}_feature_importance.png"))
            plt.close()
        elif hasattr(model, 'coef_'):
            coefs = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
            indices = np.argsort(np.abs(coefs))[::-1]
            plt.figure(figsize=(10, 6))
            sns.barplot(x=coefs[indices], y=np.array(feature_names)[indices])
            plt.title(f"{model_name} Feature Coefficients")
            plt.xlabel("Katsayı Değeri")
            plt.ylabel("Özellik")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"{model_name.replace(' ', '_').lower()}_feature_coefficients.png"))
            plt.close()
        else:
            print(f"Uyarı: Model '{model_name}' özellik önem derecelerine veya katsayılara sahip değil.")

    def plot_correlation_matrix(self, df, feature_names, title="Korelasyon Matrisi"):
        plt.figure(figsize=(12, 10))
        sns.heatmap(df[feature_names].corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "correlation_matrix.png"))
        plt.close()

    def plot_class_distribution(self, y, title="Sınıf Dağılımı"):
        plt.figure(figsize=(8, 6))
        sns.countplot(x=y)
        plt.title(title)
        plt.xlabel("Sınıf")
        plt.ylabel("Sayı")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "class_distribution_bar.png"))
        plt.close()

        plt.figure(figsize=(8, 8))
        y.value_counts().plot.pie(autopct='%1.1f%%', startangle=90, cmap="Pastel1")
        plt.title(title + " (Donut)")
        plt.ylabel('')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "class_distribution_donut.png"))
        plt.close()

    def plot_all(self, best_model, X_train, X_test, y_train, y_test, feature_names, df, model_name="Best Model"):
        print("\nTüm Görselleştirmeler Oluşturuluyor...")
        y_pred = best_model.predict(X_test)
        
        # predict_proba'nın varlığını kontrol et
        if hasattr(best_model, 'predict_proba'):
            y_proba = best_model.predict_proba(X_test)[:, 1]
        elif hasattr(best_model.named_steps['model'], 'predict_proba'):
            y_proba = best_model.named_steps['model'].predict_proba(best_model.named_steps['scaler'].transform(X_test))[:, 1]
        else:
            y_proba = None
            print("Uyarı: Modelin predict_proba metodu yok, ROC eğrisi çizilemeyecek.")


        self.plot_confusion_matrix(y_test, y_pred, model_name=model_name, normalize=False)
        self.plot_confusion_matrix(y_test, y_pred, model_name=model_name, normalize=True)
        if y_proba is not None:
            self.plot_roc_curve(y_test, y_proba, model_name=model_name)
        self.plot_feature_importance(best_model, feature_names, model_name=model_name)
        self.plot_correlation_matrix(df, feature_names + ['Outcome'], title="Veri Seti Korelasyon Matrisi")
        self.plot_class_distribution(y_test, title="Test Seti Sınıf Dağılımı")
        print("Görselleştirmeler tamamlandı ve 'results/plots' dizinine kaydedildi.")