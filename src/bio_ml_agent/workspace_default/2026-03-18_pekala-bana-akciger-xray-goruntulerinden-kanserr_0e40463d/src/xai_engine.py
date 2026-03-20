import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import os
import joblib

class XAIEngine:
    def __init__(self, model, X_train, feature_names=None, task_type="classification"):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names if feature_names is not None else [f'feature_{i}' for i in range(X_train.shape[1])]
        self.task_type = task_type
        
        if hasattr(self.model, 'named_steps') and 'scaler' in self.model.named_steps:
            self.scaler = self.model.named_steps['scaler']
            self.model_core = self.model.named_steps['model']
        else:
            self.scaler = None # No scaler in pipeline
            self.model_core = self.model

        # Ensure X_train is scaled if scaler exists for SHAP background
        if self.scaler:
            self.X_train_scaled = self.scaler.transform(X_train)
        else:
            self.X_train_scaled = X_train

        # SHAP explainer
        if hasattr(self.model_core, 'predict_proba'): # Classification
            self.explainer = shap.KernelExplainer(self.model_core.predict_proba, self.X_train_scaled)
        elif hasattr(self.model_core, 'predict'): # Regression
            self.explainer = shap.KernelExplainer(self.model_core.predict, self.X_train_scaled)
        else:
            self.explainer = None
            print("Warning: Model does not have predict_proba or predict method for SHAP explainer.")

        # LIME explainer (for tabular data, which our flattened images now are)
        if self.task_type == "classification":
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=self.X_train_scaled,
                feature_names=self.feature_names,
                class_names=['Normal', 'Pneumonia'], # Adjust based on actual class names
                mode='classification'
            )
        else:
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=self.X_train_scaled,
                feature_names=self.feature_names,
                mode='regression'
            )
        print("XAI Engine initialized.")


    def generate_shap_summary(self, X_test, output_dir="results/plots", max_display=10):
        if self.explainer is None:
            print("SHAP explainer not initialized. Skipping SHAP summary.")
            return

        os.makedirs(output_dir, exist_ok=True)
        
        # Scale X_test if a scaler is present
        if self.scaler:
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_test_scaled = X_test

        print("Generating SHAP values (this may take a while)...")
        # For KernelExplainer, calculating shap_values for all test instances can be slow for large X_test
        # We might sample X_test or just take a few instances.
        # Given our very small mock data, it should be fine.
        shap_values = self.explainer.shap_values(X_test_scaled)
        
        plt.figure(figsize=(10, 8))
        if self.task_type == "classification" and isinstance(shap_values, list): # Multi-output for classification
            # For binary classification, shap_values will be a list of two arrays.
            # We usually plot for the "positive" class (index 1).
            shap.summary_plot(shap_values[1], X_test_scaled, feature_names=self.feature_names, show=False, max_display=max_display)
        else: # Regression or single-output classification
            shap.summary_plot(shap_values, X_test_scaled, feature_names=self.feature_names, show=False, max_display=max_display)
        
        plt.title("SHAP Feature Importance Summary")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_summary_plot.png"))
        plt.close()
        print(f"SHAP summary plot saved to {os.path.join(output_dir, 'shap_summary_plot.png')}")

    def explain_instance_lime(self, instance, output_dir="results/plots", num_features=5):
        os.makedirs(output_dir, exist_ok=True)

        if self.scaler:
            instance_scaled = self.scaler.transform(instance.reshape(1, -1))[0]
        else:
            instance_scaled = instance
        
        # For LIME, the model's predict_proba or predict function needs to operate on scaled data
        # We need a wrapper function that takes unscaled data, scales it, then predicts
        def predict_fn(X_unscaled):
            if self.scaler:
                X_scaled = self.scaler.transform(X_unscaled)
            else:
                X_scaled = X_unscaled
            if hasattr(self.model_core, 'predict_proba'):
                return self.model_core.predict_proba(X_scaled)
            elif hasattr(self.model_core, 'predict'):
                return self.model_core.predict(X_scaled).reshape(-1, 1) # Ensure 2D for regression

        explanation = self.lime_explainer.explain_instance(
            data_row=instance_scaled,
            predict_fn=predict_fn,
            num_features=num_features
        )

        fig = explanation.as_pyplot_figure()
        plt.title(f"LIME Explanation for Instance (Top {num_features} Features)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "lime_explanation_plot.png"))
        plt.close()
        print(f"LIME explanation plot saved to {os.path.join(output_dir, 'lime_explanation_plot.png')}")