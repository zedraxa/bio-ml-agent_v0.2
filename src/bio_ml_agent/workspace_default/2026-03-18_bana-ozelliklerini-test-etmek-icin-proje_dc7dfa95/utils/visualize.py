import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.model_selection import learning_curve
import os

class MLVisualizer:
    def __init__(self, output_dir="results/plots"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        plt.style.use('seaborn-v0_8-darkgrid') # Use a consistent style

    def save_plot(self, fig, filename):
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Plot saved: {filepath}")

    def plot_class_distribution(self, y, title="Class Distribution", filename="class_distribution.png"):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Bar Plot
        sns.countplot(x=y, ax=axes[0], palette='viridis')
        axes[0].set_title(f'{title} (Bar Plot)')
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Count')
        for p in axes[0].patches:
            height = p.get_height()
            axes[0].annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points')

        # Donut Plot
        counts = y.value_counts()
        wedges, texts, autotexts = axes[1].pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90,
                                               pctdistance=0.85, wedgeprops=dict(width=0.3), colors=sns.color_palette('viridis', len(counts)))
        # Draw a circle at the center of the pie to make it a donut
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        fig.gca().add_artist(centre_circle)
        axes[1].set_title(f'{title} (Donut Plot)')
        axes[1].axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.

        self.save_plot(fig, filename)

    def plot_correlation_matrix(self, df, title="Correlation Matrix", filename="correlation_matrix.png"):
        fig = plt.figure(figsize=(12, 10))
        corr = df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title(title)
        self.save_plot(fig, filename)

    def plot_confusion_matrix(self, model, X_test, y_test, title="Confusion Matrix", filename="confusion_matrix.png"):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Non-normalized
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues', ax=axes[0])
        axes[0].set_title(f'{title} (Non-Normalized)')

        # Normalized
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues', normalize='true', ax=axes[1])
        axes[1].set_title(f'{title} (Normalized)')
        
        self.save_plot(fig, filename)

    def plot_roc_curve(self, model, X_test, y_test, title="ROC Curve", filename="roc_curve.png"):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if hasattr(model, 'predict_proba'):
            # For binary classification
            RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=model.__class__.__name__)
            ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
            ax.set_title(title)
            ax.legend(loc="lower right")
            self.save_plot(fig, filename)
        else:
            print("Model does not have 'predict_proba' method for ROC curve.")
            plt.close(fig) # Close the empty figure

    def plot_feature_importance(self, model, feature_names, title="Feature Importance", filename="feature_importance.png"):
        if hasattr(model, 'named_steps') and 'model' in model.named_steps:
            trained_model = model.named_steps['model']
        else:
            trained_model = model

        importances = None
        if hasattr(trained_model, 'feature_importances_'):
            importances = trained_model.feature_importances_
        elif hasattr(trained_model, 'coef_'):
            # For linear models, use absolute coefficients
            if len(trained_model.coef_.shape) > 1: # Multi-class
                importances = np.mean(np.abs(trained_model.coef_), axis=0)
            else: # Binary
                importances = np.abs(trained_model.coef_)
        
        if importances is not None:
            feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
            fig = plt.figure(figsize=(10, 8))
            sns.barplot(x=feature_importances.values, y=feature_importances.index, palette='viridis')
            plt.title(title)
            plt.xlabel("Importance")
            plt.ylabel("Feature")
            self.save_plot(fig, filename)
        else:
            print("Model does not have feature_importances_ or coef_ attribute.")

    def plot_learning_curve(self, model, X, y, title="Learning Curve", filename="learning_curve.png", cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
        fig, ax = plt.subplots(figsize=(10, 7))
        
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='accuracy' # Use accuracy as default
        )
        
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

        ax.set_title(title)
        ax.set_xlabel("Training examples")
        ax.set_ylabel("Score")
        ax.legend(loc="best")
        ax.grid(True)
        self.save_plot(fig, filename)

    def plot_all(self, model, X_train, X_test, y_train, y_test, feature_names, df):
        print("\n--- Generating Visualizations ---")
        self.plot_class_distribution(y_train, title="Training Class Distribution", filename="train_class_distribution.png")
        self.plot_class_distribution(y_test, title="Test Class Distribution", filename="test_class_distribution.png")
        self.plot_correlation_matrix(df, title="Full Dataset Correlation Matrix", filename="full_correlation_matrix.png")
        self.plot_confusion_matrix(model, X_test, y_test)
        self.plot_roc_curve(model, X_test, y_test)
        self.plot_feature_importance(model, feature_names)
        self.plot_learning_curve(model, X_train, y_train)

# Example Usage (not to be run directly by the agent, just for reference):
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=0, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    y_series = pd.Series(y, name='target')
    
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_series, test_size=0.2, random_state=42)
    
    model = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(random_state=42))])
    model.fit(X_train, y_train)
    
    viz = MLVisualizer(output_dir="temp_plots")
    viz.plot_all(model, X_train, X_test, y_train, y_test, X_df.columns, pd.concat([X_df, y_series], axis=1))