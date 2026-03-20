import shap
import lime
import lime.lime_tabular
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

class XAIEngine:
    def __init__(self, model, train_data, feature_names=None, class_names=None, task_type="classification"):
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.task_type = task_type
        
        # For traditional ML models, train_data is typically a pandas DataFrame or numpy array
        # Ensure it's not preprocessed by any scaler for LIME/SHAP background
        if isinstance(train_data, pd.DataFrame):
            self.train_data_raw = train_data.values # Use raw values for background
        else:
            self.train_data_raw = train_data # Assume numpy array

        # SHAP explainer for tree models or kernel explainer for others
        # We need to extract the actual model from the pipeline
        if hasattr(self.model, 'named_steps') and 'model' in self.model.named_steps:
            self.base_model = self.model.named_steps['model']
            # If the base model is not a tree-based model, we might need a KernelExplainer
            # For simplicity, we'll try to use a general explainer.
            # SHAP expects a prediction function.
            self.predict_fn = lambda x: self.model.predict_proba(x) if hasattr(self.model, 'predict_proba') else self.model.predict(x)
        else:
            self.base_model = self.model
            self.predict_fn = lambda x: self.model.predict_proba(x) if hasattr(self.model, 'predict_proba') else self.model.predict(x)
        
        # We assume the model object itself has a .predict_proba method for classification.
        # If not, SHAP will default to .predict which is less informative for explanations.
        
        # KernelExplainer uses a background dataset.
        # Sample a smaller background if train_data_raw is very large for performance
        if self.train_data_raw.shape[0] > 100: # Limit background size for performance
            idx = np.random.choice(self.train_data_raw.shape[0], 100, replace=False)
            self.shap_background = self.train_data_raw[idx]
        else:
            self.shap_background = self.train_data_raw

    def generate_shap_summary(self, X_test, output_dir="results/plots", max_display=10):
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating SHAP explanation for tabular data...")
        
        # For pipeline models, ensure X_test is transformed before SHAP
        if hasattr(self.model, 'named_steps') and 'scaler' in self.model.named_steps:
             # SHAP expects the background and input data to be in the same space as the model's input
             # So, if scaler is part of the pipeline, apply it to X_test
             # However, KernelExplainer works best with raw features if the model handles scaling internally.
             # Let's use the raw X_test and the pipeline's predict_proba.
             pass # Use original X_test with pipeline's predict_fn

        # Using shap.KernelExplainer for model-agnostic explanations
        explainer = shap.KernelExplainer(self.predict_fn, self.shap_background)

        # Take a subset of X_test for explanation for performance
        num_explain_samples = min(50, X_test.shape[0])
        X_test_subset = X_test.iloc[:num_explain_samples] if isinstance(X_test, pd.DataFrame) else X_test[:num_explain_samples]

        # Get SHAP values
        shap_values = explainer.shap_values(X_test_subset)
        
        # If multi-output, shap_values will be a list of arrays. For summary plot, use the first class or average.
        if isinstance(shap_values, list):
            # For multiclass, typically plot for each class or a combined view
            # Let's plot the summary for the first class, or a combined summary.
            # shap.summary_plot automatically handles multiclass output for most plotting types.
            if self.class_names:
                class_names_for_plot = self.class_names
            else:
                class_names_for_plot = [f'class_{i}' for i in range(len(shap_values))]
            
            # The summary_plot can take a list of shap_values (for multiclass)
            # The feature_names parameter is crucial for readable plots.
            shap.summary_plot(shap_values, X_test_subset, feature_names=self.feature_names,
                              class_names=class_names_for_plot, show=False)
        else:
            # Binary classification or regression
            shap.summary_plot(shap_values, X_test_subset, feature_names=self.feature_names, show=False)

        plt.savefig(os.path.join(output_dir, "shap_summary_plot.png"), bbox_inches='tight')
        plt.close()
        print(f"SHAP summary plot saved to {os.path.join(output_dir, 'shap_summary_plot.png')}")

    def explain_instance_lime(self, instance, output_dir="results/plots", num_features=5):
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating LIME explanation for a single instance...")
        
        # For LIME, the explainer needs to know the feature names.
        explainer_lime = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.shap_background, # Use the same background as SHAP
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode='classification'
        )

        # The instance should be a 1D numpy array representing a single sample.
        if isinstance(instance, pd.Series):
            instance_np = instance.values
        elif isinstance(instance, pd.DataFrame):
            instance_np = instance.iloc[0].values
        else:
            instance_np = instance

        explanation = explainer_lime.explain_instance(
            instance_np,
            self.predict_fn,
            num_features=num_features,
            top_labels=1
        )
        
        # Save the LIME explanation plot
        fig = explanation.as_pyplot_figure()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "lime_explanation_instance.png"), bbox_inches='tight')
        plt.close(fig)
        
        print(f"LIME explanation plot saved to {os.path.join(output_dir, 'lime_explanation_instance.png')}")
        print("\nLIME Explanation Details:")
        for x in explanation.as_list():
            print(x)