import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay
from sklearn.preprocessing import label_binarize
import os

class MLVisualizer:
    def __init__(self, output_dir="results/plots"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_confusion_matrix(self, y_true, y_pred, class_names, normalize=False, title="Confusion Matrix"):
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = "Normalized " + title
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(os.path.join(self.output_dir, f"{'normalized_' if normalize else ''}confusion_matrix.png"))
        plt.close()
        print(f"Plotting {'Normalized ' if normalize else ''}Confusion Matrix...")
        print(f"{'Normalized ' if normalize else ''}confusion matrix saved to {os.path.join(self.output_dir, f'{'normalized_' if normalize else ''}confusion_matrix.png')}")

    def plot_roc_curve(self, y_true, y_prob, class_names, task_type="classification"):
        plt.figure(figsize=(10, 8))
        if task_type == "binary":
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                     label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
        elif task_type == "classification" and len(class_names) > 2: # Multiclass OvR
            # Binarize the true labels
            y_true_binarized = label_binarize(y_true, classes=np.unique(y_true))
            n_classes = len(np.unique(y_true))

            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_prob[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            plt.plot([0, 1], [0, 1], 'k--')
            for i in range(n_classes):
                plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:.2f})')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Multiclass Receiver Operating Characteristic (ROC) Curve (One-vs-Rest)')
            plt.legend(loc="lower right")
        else: # Binary or other (should be handled by binary case)
            print("ROC curve not applicable or implemented for this task type/number of classes.")
            return

        plt.savefig(os.path.join(self.output_dir, "roc_curve.png"))
        plt.close()
        print(f"Plotting {task_type.capitalize()} ROC Curve...")
        print(f"ROC curve saved to {os.path.join(self.output_dir, 'roc_curve.png')}")

    def plot_class_distribution(self, y, class_names, title="Class Distribution"):
        plt.figure(figsize=(8, 6))
        sns.countplot(x=y)
        plt.xticks(ticks=np.arange(len(class_names)), labels=class_names, rotation=45, ha='right')
        plt.title(title)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.savefig(os.path.join(self.output_dir, "class_distribution.png"))
        plt.close()
        print(f"Class distribution plot saved to {os.path.join(self.output_dir, 'class_distribution.png')}")

    def plot_feature_importance(self, model, feature_names, title="Feature Importance"):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)[0] # For binary classification, for multiclass might need more logic
            if importances.ndim > 1: # For multiclass linear models
                importances = np.mean(np.abs(model.coef_), axis=0) # Average abs coefficients
        else:
            print("Model does not have feature_importances_ or coef_ attribute.")
            return

        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title(title)
        sns.barplot(x=importances[indices][:10], y=np.array(feature_names)[indices][:10]) # Top 10 features
        plt.xlabel("Relative Importance")
        plt.ylabel("Feature Name")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "feature_importance.png"))
        plt.close()
        print(f"Feature importance plot saved to {os.path.join(self.output_dir, 'feature_importance.png')}")