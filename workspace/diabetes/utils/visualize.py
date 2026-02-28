import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import learning_curve
import os

class MLVisualizer:
    def __init__(self, output_dir="results/plots"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_confusion_matrix(self, y_true, y_pred, labels=None, normalize=False, title="Confusion Matrix"):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title += " (Normalized)"
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues", cbar=False,
                    xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, title.replace(" ", "_").replace("(", "").replace(")", "") + ".png"))
        plt.close()

    def plot_roc_curve(self, model, X_test, y_test, title="ROC Curve"):
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(title)
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, title.replace(" ", "_") + ".png"))
            plt.close()
        else:
            print("Modelin predict_proba metodu yok, ROC Curve çizilemiyor.")

    def plot_feature_importance(self, model_pipeline, feature_names, title="Feature Importance"):
        model = model_pipeline.named_steps['model']
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
        else:
            print("Modelin özellik önemini gösteren bir özniteliği yok (feature_importances_ veya coef_).")
            return

        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, title.replace(" ", "_") + ".png"))
        plt.close()

    def plot_correlation_matrix(self, df, title="Correlation Matrix"):
        plt.figure(figsize=(12, 10))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, title.replace(" ", "_") + ".png"))
        plt.close()

    def plot_learning_curve(self, model_pipeline, X, y, title="Learning Curve", cv=5, n_jobs=-1):
        train_sizes, train_scores, test_scores = learning_curve(
            model_pipeline, X, y, cv=cv, n_jobs=n_jobs, train_sizes=np.linspace(.1, 1.0, 5), scoring='roc_auc'
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        plt.title(title)
        plt.xlabel("Training examples")
        plt.ylabel("Score (ROC AUC)")
        plt.legend(loc="best")
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, title.replace(" ", "_") + ".png"))
        plt.close()
        
    def plot_class_distribution(self, y, title="Class Distribution"):
        plt.figure(figsize=(8, 6))
        y.value_counts().plot(kind='bar')
        plt.title(title)
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, title.replace(" ", "_") + ".png"))
        plt.close()

        plt.figure(figsize=(8, 6))
        y.value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, cmap='Pastel1')
        plt.title(title + " (Pie Chart)")
        plt.ylabel('') # Hide the default 'y' label for pie chart
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, title.replace(" ", "_") + "_pie.png"))
        plt.close()


    def plot_all(self, model_pipeline, X_train, X_test, y_train, y_test, feature_names, df, model_name="Best Model", task_type="classification", target_name='Outcome'):
        print(f"'{model_name}' için görselleştirmeler oluşturuluyor...")
        
        y_pred_test = model_pipeline.predict(X_test)
        class_labels = y_train.unique()
        if len(class_labels) == 2: # Binary classification
            class_labels = sorted(class_labels) # Ensure order 0, 1

        self.plot_confusion_matrix(y_test, y_pred_test, labels=class_labels, title=f"{model_name} Confusion Matrix")
        self.plot_confusion_matrix(y_test, y_pred_test, labels=class_labels, normalize=True, title=f"{model_name} Normalized Confusion Matrix")
        
        if task_type == "classification":
            self.plot_roc_curve(model_pipeline, X_test, y_test, title=f"{model_name} ROC Curve")
            
        self.plot_feature_importance(model_pipeline, feature_names, title=f"{model_name} Feature Importance")
        self.plot_correlation_matrix(df.drop(columns=[target_name], errors='ignore'), title="Overall Feature Correlation Matrix")
        self.plot_learning_curve(model_pipeline, X_train, y_train, title=f"{model_name} Learning Curve")
        self.plot_class_distribution(df[target_name], title="Target Class Distribution")
        print("Tüm görselleştirmeler tamamlandı.")