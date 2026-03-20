import shap
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from lime import lime_tabular
from lime.lime_tabular import LimeTabularExplainer

class XAIEngine:
    def __init__(self, model, X_train, feature_names, task_type="classification", class_names=None):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.task_type = task_type
        self.class_names = class_names

        # Ensure X_train is a DataFrame for SHAP, especially for pipeline.
        if not isinstance(self.X_train, pd.DataFrame):
            self.X_train_df = pd.DataFrame(self.X_train, columns=self.feature_names)
        else:
            self.X_train_df = self.X_train

        # For SHAP, if the model is a pipeline, we need to apply scaler to X_train for background data
        # but the predict/predict_proba methods of the pipeline handle scaling internally for explanation.
        # So explainer should wrap the full pipeline.
        # The background data needs to be preprocessed by the pipeline's scaler before passing to KernelExplainer,
        # or we use an explainer that works directly with the pipeline (like TreeExplainer for tree models).
        
        # Check if the model within the pipeline is a tree-based model for TreeExplainer
        self.is_tree_model = False
        if hasattr(model, 'named_steps') and 'model' in model.named_steps:
            inner_model = model.named_steps['model']
            if any(isinstance(inner_model, cls) for cls in [
                shap.TreeExplainer.model_type_mapping['xgboost.core.Booster'],
                shap.TreeExplainer.model_type_mapping['lightgbm.basic.Booster'],
                shap.TreeExplainer.model_type_mapping['sklearn.ensemble.RandomForestRegressor'],
                shap.TreeExplainer.model_type_mapping['sklearn.ensemble.RandomForestClassifier'],
                shap.TreeExplainer.model_type_mapping['sklearn.ensemble.GradientBoostingRegressor'],
                shap.TreeExplainer.model_type_mapping['sklearn.ensemble.GradientBoostingClassifier'],
                shap.TreeExplainer.model_type_mapping['sklearn.tree.DecisionTreeRegressor'],
                shap.TreeExplainer.model_type_mapping['sklearn.tree.DecisionTreeClassifier']
            ]):
                self.is_tree_model = True
        
        # Determine appropriate SHAP explainer
        if self.is_tree_model:
            self.explainer = shap.TreeExplainer(self.model.named_steps['model'], data=self.model.named_steps['scaler'].transform(self.X_train_df), feature_names=self.feature_names)
        else:
            # Use KernelExplainer for general models (slower)
            # Sample background data for KernelExplainer for performance
            background_data = shap.sample(self.model.named_steps['scaler'].transform(self.X_train_df), 100)
            self.explainer = shap.KernelExplainer(self.model.predict_proba if self.task_type == "classification" else self.model.predict,
                                                 background_data,
                                                 feature_names=self.feature_names)
            
    def generate_shap_summary(self, X_test, output_dir="results/plots", max_display=10, filename="shap_summary.png"):
        os.makedirs(output_dir, exist_ok=True)
        print("SHAP özet grafiği oluşturuluyor...")

        # For SHAP, X_test also needs to be transformed if the explainer was fitted on transformed data
        # However, if self.model is a pipeline, shap.Explainer can often handle it if predict_proba is callable.
        # Let's assume explainer's `shap_values` method is smart enough with pipelines.
        # If explainer was TreeExplainer and fitted on inner model with scaled data, X_test needs to be scaled.
        if self.is_tree_model:
            X_test_transformed = self.model.named_steps['scaler'].transform(X_test)
            shap_values = self.explainer.shap_values(X_test_transformed)
        else:
            # KernelExplainer should work with raw X_test and the pipeline's predict_proba/predict
            shap_values = self.explainer.shap_values(X_test)

        if self.task_type == "classification" and isinstance(shap_values, list):
            # For multi-class classification, plot for one class (e.g., the positive class if binary)
            # Assuming binary classification, we care about the positive class (index 1)
            # Or for multi-class, it depends on which class's explanation is desired.
            # For a general summary, the average magnitude of SHAP values across classes can be used,
            # but usually, we pick one class to explain.
            # Here, let's pick the second class if it's a list (common for binary classification, shap_values[1] for positive class)
            if len(shap_values) == 2: # Binary classification
                shap.summary_plot(shap_values[1], X_test, feature_names=self.feature_names, show=False, max_display=max_display)
            else: # Multi-class, take mean or first class for summary
                shap.summary_plot(shap_values[0], X_test, feature_names=self.feature_names, show=False, max_display=max_display)
        else: # Regression or single output classification
            shap.summary_plot(shap_values, X_test, feature_names=self.feature_names, show=False, max_display=max_display)
        
        plt.title('SHAP Özellik Önem Özeti')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        print(f"SHAP özet grafiği kaydedildi: {os.path.join(output_dir, filename)}")
        
        # Get top features from SHAP values for the report
        if self.task_type == "classification" and isinstance(shap_values, list):
             # For binary, use positive class, for multi, use average abs or first class
            if len(shap_values) == 2:
                abs_shap_values = np.abs(shap_values[1])
            else:
                abs_shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0) # Mean across classes
        else:
            abs_shap_values = np.abs(shap_values)
            
        mean_abs_shap = np.mean(abs_shap_values, axis=0)
        shap_feature_importance = pd.DataFrame({'feature': self.feature_names, 'shap_importance': mean_abs_shap})
        shap_feature_importance = shap_feature_importance.sort_values(by='shap_importance', ascending=False)
        return shap_feature_importance.head(max_display)


    def explain_instance_lime(self, instance, output_dir="results/plots", filename="lime_explanation.png"):
        os.makedirs(output_dir, exist_ok=True)
        print(f"LIME tekil örnek açıklaması oluşturuluyor (Örnek: {instance.name})...")

        # LIME explainer needs the raw (unscaled) training data for background statistics
        # and the model's predict_proba method.
        # If the model is a pipeline, its predict_proba naturally handles the scaling.
        
        # Ensure instance is a DataFrame row for consistent access
        if isinstance(instance, pd.Series):
            instance_df = pd.DataFrame([instance], columns=self.feature_names)
        else: # Assume numpy array
            instance_df = pd.DataFrame([instance], columns=self.feature_names)


        # Lime explainer expects a prediction function that takes a numpy array
        # and returns probabilities for each class.
        # It also expects training data as a numpy array, but for numerical features.
        # It needs to know which features are categorical.

        # Assume all features are numerical for simplicity, based on typical tabular data.
        # If there are categorical features, define 'categorical_features' and 'categorical_names'
        # The training data for LIME explainer should be the original, unscaled numerical data for proper perturbation.
        # But the model expects scaled data. LIME will perturb unscaled data then internally scale it for prediction.

        explainer = LimeTabularExplainer(
            training_data=self.X_train_df.values,
            feature_names=self.feature_names,
            class_names=self.class_names, # Use class_names for classification
            mode=self.task_type
        )
        
        # The model's predict_proba (or predict for regression) is passed directly.
        # The explainer will pass perturbed (but preprocessed by Lime) data to this function.
        # If self.model is a Pipeline, it already contains the scaler.
        explanation = explainer.explain_instance(
            data_row=instance_df.values[0], # Pass the single instance as a numpy array
            predict_fn=self.model.predict_proba if self.task_type == "classification" else self.model.predict,
            num_features=10
        )

        fig = explanation.as_pyplot_figure()
        plt.title(f'LIME Açıklaması - Örnek {instance.name}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        print(f"LIME açıklaması kaydedildi: {os.path.join(output_dir, filename)}")