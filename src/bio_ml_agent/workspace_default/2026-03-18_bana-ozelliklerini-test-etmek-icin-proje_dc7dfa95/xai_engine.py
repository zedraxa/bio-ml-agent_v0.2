import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

class XAIEngine:
    def __init__(self, model, X_train_for_explainer, feature_names, task_type="classification"):
        """
        Initializes the XAI Engine.

        Args:
            model: The trained model (can be a scikit-learn pipeline).
            X_train_for_explainer (pd.DataFrame): Training data used to fit the explainer (e.g., background data for SHAP).
            feature_names (list): List of feature names.
            task_type (str): "classification" or "regression".
        """
        self.model = model
        self.feature_names = feature_names
        self.task_type = task_type
        
        # Extract the actual model if it's a pipeline
        if hasattr(self.model, 'named_steps') and 'model' in self.model.named_steps:
            self.trained_model = self.model.named_steps['model']
            self.scaler = self.model.named_steps['scaler'] if 'scaler' in self.model.named_steps else None
        else:
            self.trained_model = self.model
            self.scaler = None

        # Prepare background data for SHAP, scaled if a scaler is present
        if self.scaler:
            self.X_train_scaled = self.scaler.transform(X_train_for_explainer)
        else:
            self.X_train_scaled = X_train_for_explainer.values # Convert DataFrame to numpy array for shap

        # SHAP explainer
        self.explainer = self._get_shap_explainer()

        # LIME explainer
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.X_train_scaled,
            feature_names=self.feature_names,
            class_names=['No Diabetes', 'Diabetes'] if self.task_type == "classification" else ['Target'],
            mode=self.task_type
        )

    def _get_shap_explainer(self):
        # Choose SHAP explainer based on model type
        if "XGB" in str(type(self.trained_model)) or "LGBM" in str(type(self.trained_model)) or "CatBoost" in str(type(self.trained_model)):
            return shap.TreeExplainer(self.trained_model)
        elif "Linear" in str(type(self.trained_model)) or "Logistic" in str(type(self.trained_model)):
            return shap.LinearExplainer(self.trained_model, self.X_train_scaled)
        else: # KernelExplainer for other models (e.g., RandomForest, SVM) - can be slow for large datasets
            # For KernelExplainer, use a subset of the background data for speed
            if self.X_train_scaled.shape[0] > 100:
                background = shap.sample(self.X_train_scaled, 100)
            else:
                background = self.X_train_scaled
            
            # Predict function for KernelExplainer
            if self.task_type == "classification":
                if hasattr(self.trained_model, 'predict_proba'):
                    predict_fn = lambda x: self.trained_model.predict_proba(x)
                else: # Fallback for models without predict_proba, but it's not ideal for classification SHAP
                    predict_fn = lambda x: np.array([[0,1] if p==1 else [1,0] for p in self.trained_model.predict(x)]) # dummy proba
            else:
                predict_fn = self.trained_model.predict

            return shap.KernelExplainer(predict_fn, background)

    def _predict_fn_for_lime(self, X_input):
        """Wrapper predict function for LIME, handles scaling if present."""
        if self.scaler:
            X_input_scaled = self.scaler.transform(X_input)
            return self.trained_model.predict_proba(X_input_scaled)
        else:
            return self.trained_model.predict_proba(X_input)

    def generate_shap_summary(self, X_data_to_explain, output_dir="results/plots", max_display=10, filename="shap_summary_plot.png"):
        """
        Generates and saves a SHAP summary plot.

        Args:
            X_data_to_explain (pd.DataFrame): Data for which to generate SHAP values.
            output_dir (str): Directory to save the plot.
            max_display (int): Maximum number of features to display.
            filename (str): Name of the output file.
        """
        print(f"Generating SHAP summary plot for {X_data_to_explain.shape[0]} instances...")
        if self.scaler:
            X_data_scaled = self.scaler.transform(X_data_to_explain)
        else:
            X_data_scaled = X_data_to_explain.values

        shap_values = self.explainer.shap_values(X_data_scaled)

        # For classification, shap_values will be a list of arrays (one for each class).
        # We usually visualize for the positive class (index 1).
        if self.task_type == "classification" and isinstance(shap_values, list):
            shap_values = shap_values[1] # SHAP values for the positive class (e.g., diabetes)

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_data_scaled, feature_names=self.feature_names, max_display=max_display, show=False)
        
        filepath = os.path.join(output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"SHAP summary plot saved to {filepath}")

    def explain_instance_lime(self, instance, output_dir="results/plots", num_features=10, filename="lime_explanation_instance.png"):
        """
        Generates and saves a LIME explanation for a single instance.

        Args:
            instance (pd.Series): A single data instance (row from X_test) to explain.
            output_dir (str): Directory to save the plot.
            num_features (int): Number of features to show in the LIME explanation.
            filename (str): Name of the output file.
        """
        print(f"Generating LIME explanation for instance: {instance.name}")
        
        # LIME expects a 2D array, even for a single instance
        instance_values = instance.values.reshape(1, -1)

        if self.task_type == "classification":
            explanation = self.lime_explainer.explain_instance(
                data_row=instance_values[0], 
                predict_fn=self._predict_fn_for_lime,
                num_features=num_features,
                labels=(0, 1) # Explicitly define labels for classification
            )
        else:
            explanation = self.lime_explainer.explain_instance(
                data_row=instance_values[0], 
                predict_fn=self._predict_fn_for_lime,
                num_features=num_features
            )

        fig = explanation.as_pyplot_figure()
        plt.title(f"LIME Explanation for Instance {instance.name}")
        filepath = os.path.join(output_dir, filename)
        fig.tight_layout()
        fig.savefig(filepath, dpi=300)
        plt.close(fig)
        print(f"LIME explanation plot saved to {filepath}")

# Example Usage (not to be run directly by the agent, just for reference):
if __name__ == '__main__':
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd

    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=0, random_state=42, n_classes=2)
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')

    X_train, X_test, y_train, y_test = train_test_split(X_df, y_series, test_size=0.2, random_state=42, stratify=y_series)

    # Train a RandomForest model within a pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(random_state=42))
    ])
    pipeline.fit(X_train, y_train)

    # Initialize XAI Engine
    xai_engine = XAIEngine(pipeline, X_train, feature_names=feature_names, task_type="classification")

    # Generate SHAP summary plot
    xai_engine.generate_shap_summary(X_test, output_dir="temp_xai_plots")

    # Generate LIME explanation for a specific instance
    xai_engine.explain_instance_lime(X_test.iloc[0], output_dir="temp_xai_plots", filename="lime_instance_0.png")