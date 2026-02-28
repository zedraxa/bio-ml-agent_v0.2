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
        os.makedirs(output_dir, exist_ok=True)

    def plot_confusion_matrix(self, y_true, y_pred, labels=None, normalize=False, title="Confusion Matrix", filename="confusion_matrix.png"):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = "Normalized " + title
            filename = "normalized_" + filename
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        print(f"Grafik kaydedildi: {os.path.join(self.output_dir, filename)}")

    def plot_roc_curve(self, model, X_test, y_test, title="ROC Curve", filename="roc_curve.png", task_type="classification"):
        if task_type != "classification":
            print("ROC eğrisi sadece sınıflandırma görevleri için uygundur.")
            return

        plt.figure(figsize=(8, 6))
        
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
            # For binary classification, decision_function returns (n_samples,)
            # We need to convert it to (n_samples, 2) for roc_curve when dealing with OvR
            if y_score.ndim == 1:
                y_score = np.vstack([-y_score, y_score]).T # Simple conversion for OvR logic
        else:
            print("Modelin predict_proba veya decision_function özelliği yok. ROC eğrisi çizilemiyor.")
            return

        if y_score.ndim == 1: # Binary case where decision_function returns a 1D array
             fpr, tpr, _ = roc_curve(y_test, y_score)
             roc_auc = auc(fpr, tpr)
             plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        elif y_score.shape[1] == 2: # Binary classification with predict_proba
            fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        else: # Multi-class (One-vs-Rest)
            n_classes = y_score.shape[1]
            for i in range(n_classes):
                y_test_bin = (y_test == model.classes_[i]).astype(int)
                fpr, tpr, _ = roc_curve(y_test_bin, y_score[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'ROC curve of class {model.classes_[i]} (area = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        print(f"Grafik kaydedildi: {os.path.join(self.output_dir, filename)}")

    def plot_feature_importance(self, model, feature_names, title="Feature Importance", filename="feature_importance.png"):
        plt.figure(figsize=(10, 7))
        
        # Check if the model is a pipeline and get the actual model
        if hasattr(model, 'named_steps') and 'model' in model.named_steps:
            actual_model = model.named_steps['model']
        else:
            actual_model = model

        importances = None
        if hasattr(actual_model, 'feature_importances_'):
            importances = actual_model.feature_importances_
        elif hasattr(actual_model, 'coef_'):
            importances = np.abs(actual_model.coef_)
            if importances.ndim > 1: # For multi-class, take mean of absolute coefficients
                importances = np.mean(importances, axis=0)

        if importances is not None:
            df_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})
            df_importance = df_importance.sort_values('importance', ascending=False)
            sns.barplot(x='importance', y='feature', data=df_importance)
            plt.title(title)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, filename))
            plt.close()
            print(f"Grafik kaydedildi: {os.path.join(self.output_dir, filename)}")
        else:
            print("Modelde feature_importances_ veya coef_ özelliği bulunamadı.")

    def plot_correlation_matrix(self, df, title="Correlation Matrix", filename="correlation_matrix.png"):
        plt.figure(figsize=(12, 10))
        sns.heatmap(df.corr(), annot=False, cmap='coolwarm', fmt=".2f")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        print(f"Grafik kaydedildi: {os.path.join(self.output_dir, filename)}")

    def plot_learning_curve(self, model, X, y, title="Learning Curve", filename="learning_curve.png", cv=5, n_jobs=-1):
        plt.figure(figsize=(10, 7))
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=n_jobs, train_sizes=np.linspace(.1, 1.0, 5)
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.grid()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.title(title)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        print(f"Grafik kaydedildi: {os.path.join(self.output_dir, filename)}")
    
    def plot_class_distribution(self, y, title="Class Distribution", filename_bar="class_distribution_bar.png", filename_donut="class_distribution_donut.png"):
        class_counts = pd.Series(y).value_counts()

        # Bar Plot
        plt.figure(figsize=(8, 6))
        sns.barplot(x=class_counts.index, y=class_counts.values)
        plt.title(f'{title} (Bar Plot)')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename_bar))
        plt.close()
        print(f"Grafik kaydedildi: {os.path.join(self.output_dir, filename_bar)}")

        # Donut Plot
        plt.figure(figsize=(8, 8))
        plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.3))
        centre_circle = plt.Circle((0,0),0.70,fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        plt.title(f'{title} (Donut Plot)')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename_donut))
        plt.close()
        print(f"Grafik kaydedildi: {os.path.join(self.output_dir, filename_donut)}")

    def plot_all(self, model, X_train, X_test, y_train, y_test, feature_names=None, df=None, task_type="classification"):
        print("Tüm görselleştirmeler oluşturuluyor...")
        y_pred = model.predict(X_test)
        labels = np.unique(y_test)
        
        self.plot_confusion_matrix(y_test, y_pred, labels=labels, normalize=False, title="Confusion Matrix")
        self.plot_confusion_matrix(y_test, y_pred, labels=labels, normalize=True, title="Normalized Confusion Matrix")
        
        if task_type == "classification":
            self.plot_roc_curve(model, X_test, y_test, task_type=task_type)

        if feature_names is not None:
            self.plot_feature_importance(model, feature_names)

        if df is not None:
            self.plot_correlation_matrix(df)

        self.plot_learning_curve(model, X_train, y_train)
        self.plot_class_distribution(y_train)