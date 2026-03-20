import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.inspection import permutation_importance

class MLVisualizer:
    def __init__(self, output_dir="results/plots"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Set a consistent and professional style for all plots
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("viridis")

    def _save_plot(self, fig, filename):
        """Saves the figure to the output directory."""
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Plot saved to: {path}")

    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """Plots and saves both normalized and non-normalized confusion matrices."""
        # Non-normalized confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        self._save_plot(fig, 'confusion_matrix.png')

        # Normalized confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig_norm, ax_norm = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax_norm)
        ax_norm.set_title('Normalized Confusion Matrix')
        ax_norm.set_ylabel('True Label')
        ax_norm.set_xlabel('Predicted Label')
        self._save_plot(fig_norm, 'confusion_matrix_normalized.png')

    def plot_roc_curve(self, y_true, y_proba):
        """Plots and saves the ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        self._save_plot(fig, 'roc_curve.png')

    def plot_feature_importance(self, model, X, feature_names):
        """Plots and saves feature importances based on model coefficients or Gini importance."""
        if hasattr(model.named_steps['model'], 'coef_'):
            # For linear models (e.g., Logistic Regression)
            importances = np.abs(model.named_steps['model'].coef_[0])
        elif hasattr(model.named_steps['model'], 'feature_importances_'):
            # For tree-based models (e.g., RandomForest)
            importances = model.named_steps['model'].feature_importances_
        else:
            print("Model does not have 'coef_' or 'feature_importances_'. Skipping feature importance plot.")
            return

        indices = np.argsort(importances)[::-1]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title("Feature Importances")
        sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], ax=ax)
        ax.set_xlabel("Importance")
        ax.set_ylabel("Features")
        self._save_plot(fig, 'feature_importance.png')
        
    def plot_correlation_matrix(self, df):
        """Plots and saves the correlation matrix of the dataframe."""
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax)
        ax.set_title('Feature Correlation Matrix')
        self._save_plot(fig, 'correlation_matrix.png')
        
    def plot_all(self, model, X_test, y_test, feature_names, df):
        """Runs all visualization methods for a given model and data."""
        print("\n--- Generating Visualizations ---")
        y_pred = model.predict(X_test)
        
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            self.plot_roc_curve(y_test, y_proba)
        else:
            print("Model cannot produce probabilities, skipping ROC curve.")

        self.plot_confusion_matrix(y_test, y_pred, class_names=['Healthy', 'Disease'])
        self.plot_feature_importance(model, X_test, feature_names)
        self.plot_correlation_matrix(df)
        print("--- Visualizations Generated ---")