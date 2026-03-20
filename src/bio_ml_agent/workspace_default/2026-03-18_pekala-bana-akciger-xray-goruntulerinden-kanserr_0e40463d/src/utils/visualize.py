import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
import os

class MLVisualizer:
    def __init__(self, output_dir="results/plots"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_confusion_matrix(self, y_true, y_pred, class_names, normalize=False, title='Confusion Matrix', filename='confusion_matrix.png'):
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized ' + title
            fmt = '.2f'
        else:
            fmt = 'd'

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

    def plot_roc_curve(self, y_true, y_proba, class_names, title='ROC Curve', filename='roc_curve.png'):
        if len(class_names) == 2: # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
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
            plt.savefig(os.path.join(self.output_dir, filename))
            plt.close()
        else: # Multi-class (OvR)
            print("Multi-class ROC not implemented in this version of plot_roc_curve directly. Please provide one-hot encoded y_true and probabilities for each class for OvR.")
            # Example of how it would be done (requires one-hot encoding y_true and getting per-class y_proba)
            # from sklearn.preprocessing import label_binarize
            # y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
            # for i, class_name in enumerate(class_names):
            #    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            #    roc_auc = auc(fpr, tpr)
            #    plt.plot(fpr, tpr, lw=2, label=f'ROC curve of class {class_name} (area = {roc_auc:.2f})')
            # ... and so on

    def plot_feature_importance(self, model, feature_names, top_n=10, title='Feature Importance', filename='feature_importance.png'):
        if hasattr(model, 'named_steps') and 'model' in model.named_steps:
            actual_model = model.named_steps['model']
        else:
            actual_model = model # Assume model is directly the classifier

        importances = None
        if hasattr(actual_model, 'feature_importances_'):
            importances = actual_model.feature_importances_
        elif hasattr(actual_model, 'coef_'):
            if actual_model.coef_.ndim > 1: # Multi-class, take mean of absolute coeffs
                importances = np.mean(np.abs(actual_model.coef_), axis=0)
            else: # Binary class
                importances = np.abs(actual_model.coef_)
        
        if importances is not None:
            indices = np.argsort(importances)[::-1]
            top_indices = indices[:top_n]
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x=importances[top_indices], y=[feature_names[i] for i in top_indices], palette='viridis')
            plt.title(title)
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, filename))
            plt.close()
        else:
            print(f"Model {type(actual_model).__name__} does not have feature importances or coefficients.")

    def plot_all(self, model, X_train, X_test, y_train, y_test, feature_names=None, df=None, class_names=None, is_image_classification=False):
        # Determine class names
        if class_names is None:
            class_names = [str(c) for c in np.unique(y_train)]
        
        # Predict on test set
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_test)
        else:
            print("Model does not have a 'predict' method. Cannot generate predictions for plots.")
            return

        # Predict probabilities if available for ROC curve
        y_proba = None
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
        
        self.plot_confusion_matrix(y_test, y_pred, class_names, normalize=False, filename='confusion_matrix_raw.png')
        self.plot_confusion_matrix(y_test, y_pred, class_names, normalize=True, filename='confusion_matrix_normalized.png')

        if y_proba is not None and len(class_names) == 2: # ROC for binary
            self.plot_roc_curve(y_test, y_proba, class_names, filename='roc_curve.png')
        elif y_proba is not None and len(class_names) > 2:
             print("Multi-class ROC curve plotting is currently simplified, consider custom implementation.")
             # For a simplified multi-class ROC, we can plot one vs rest for each class
             for i, class_name in enumerate(class_names):
                 y_true_binary = (y_test == i).astype(int)
                 self.plot_roc_curve(y_true_binary, y_proba, [f"Not {class_name}", class_name], title=f'ROC Curve (OvR) for {class_name}', filename=f'roc_curve_ovr_{class_name}.png')

        if feature_names is not None and not is_image_classification: # Feature importance for tabular data
            self.plot_feature_importance(model, feature_names)
        elif is_image_classification:
            print("Feature importance for image classification is typically derived from CNN activations (e.g. Grad-CAM), not directly from pixel features in this generic plotter.")
            # For image models, we skip pixel-level feature importance here
            pass
        
        # Additional plots can be added here (e.g., Learning Curve, Class Distribution, Correlation Matrix)
        # However, for mock data, these might not be meaningful or require more robust data.
        print(f"All visualizations saved to {self.output_dir}")