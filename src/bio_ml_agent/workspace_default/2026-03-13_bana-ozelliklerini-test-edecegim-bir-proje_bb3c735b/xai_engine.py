import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

class XAIEngine:
    def __init__(self, model_pipeline, X_train_df, feature_names, task_type="classification"):
        self.model_pipeline = model_pipeline
        self.X_train_df = X_train_df
        self.feature_names = feature_names
        self.task_type = task_type
        
        # Predict fonksiyonunu pipeline'dan al
        if hasattr(model_pipeline, 'predict_proba'):
            self.predict_fn = model_pipeline.predict_proba
        elif hasattr(model_pipeline.named_steps['model'], 'predict_proba'):
            # Eğer model pipeline içinde ise ve predict_proba'sı varsa
            # Bu durumda X'i scaler'dan geçirmek gerekir
            self.predict_fn = lambda x: model_pipeline.predict_proba(pd.DataFrame(x, columns=self.feature_names))
        else:
            raise ValueError("Modelin predict_proba metodu yok. Sınıflandırma için bu metot gereklidir.")

    def generate_shap_summary(self, X_test_df, output_dir="results/plots", max_display=10):
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            model = self.model_pipeline.named_steps['model']
            scaler = self.model_pipeline.named_steps['scaler']
            X_test_scaled = scaler.transform(X_test_df)
            X_train_scaled = scaler.transform(self.X_train_df) # KernelExplainer ve LinearExplainer için

            shap_values = None
            
            # Model tipine göre SHAP explainer seçimi
            if hasattr(model, 'tree_'): # Ağaç tabanlı modeller için (RandomForest, GradientBoosting)
                # TreeExplainer'ı doğrudan modelle başlat
                explainer = shap.TreeExplainer(model)
                # SHAP değerlerini ölçeklendirilmiş test verisi üzerinde hesapla
                shap_values = explainer.shap_values(X_test_scaled)
                # shap.summary_plot'a X_test_df (orijinal özellik isimleri için) ve feature_names ver
                plot_data = X_test_df
            elif hasattr(model, 'coef_'): # Doğrusal modeller için (LogisticRegression)
                explainer = shap.LinearExplainer(model, X_train_scaled)
                shap_values = explainer.shap_values(X_test_scaled)
                plot_data = X_test_df
            else:
                # Diğer modeller için KernelExplainer (daha yavaş olabilir)
                # KernelExplainer'a doğrudan pipeline'ın predict_proba'sını ve orijinal X_train_df'i veriyoruz
                # Böylece Pipeline kendi içindeki scaler'ı kullanabilir.
                # Eğitim verisinin küçük bir örneğini kullanmak performans için önemlidir.
                explainer = shap.KernelExplainer(self.model_pipeline.predict_proba, shap.utils.sample(self.X_train_df, 100))
                shap_values = explainer.shap_values(X_test_df) # KernelExplainer'a orijinal X_test_df veriyoruz
                plot_data = X_test_df # Orijinal X_test_df'i kullanmaya devam

            if shap_values is None:
                raise ValueError("SHAP değerleri hesaplanamadı.")

            # Sınıflandırma durumunda, pozitif sınıfın SHAP değerlerini al
            if self.task_type == "classification" and isinstance(shap_values, list):
                shap_values = shap_values[1] # For the positive class (assuming binary classification)

            plt.figure(figsize=(10, 8))
            # shap.summary_plot'a shap_values'u ve orijinal plot_data'yı vermemiz gerekiyor.
            # feature_names parametresi de belirtilirse, plot_data'nın sütun isimleri yerine bu kullanılabilir.
            shap.summary_plot(shap_values, plot_data, feature_names=self.feature_names, show=False, max_display=max_display)
            plt.title(f"{self.model_pipeline.named_steps['model'].__class__.__name__} SHAP Summary Plot")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "shap_summary_plot.png"))
            plt.close()
            print("SHAP Summary Plot oluşturuldu ve kaydedildi.")
            
            # En önemli özellikleri belirle
            abs_shap_means = np.abs(shap_values).mean(axis=0)
            sorted_indices = np.argsort(abs_shap_means)[::-1]
            top_features = [(self.feature_names[i], abs_shap_means[i]) for i in sorted_indices[:max_display]]
            return top_features

        except Exception as e:
            print(f"SHAP oluşturulurken hata oluştu: {e}. Lütfen SHAP ile uyumluluğu kontrol edin veya farklı bir explainer deneyin.")
            return []

    def explain_instance_lime(self, instance_df, output_dir="results/plots", num_features=5, filename="lime_explanation_instance.png"):
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # LIME explainer
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=self.X_train_df.values, # Orijinal eğitim verisi
                feature_names=self.feature_names,
                class_names=["Negative", "Positive"], # Varsayılan olarak ikili sınıflandırma
                mode=self.task_type
            )

            # Instance'ı numpy array'e dönüştür
            instance_array = instance_df.values.reshape(1, -1)[0]
            
            # Predict fonksiyonunu LIME'a uygun hale getir
            def lime_predict_fn(data):
                df_data = pd.DataFrame(data, columns=self.feature_names)
                return self.predict_fn(df_data)

            exp = lime_explainer.explain_instance(
                data_row=instance_array,
                predict_fn=lime_predict_fn,
                num_features=num_features
            )
            
            # LIME görselleştirmesi
            fig = exp.as_pyplot_figure()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, filename))
            plt.close(fig)
            print(f"LIME Açıklaması ('{filename}') oluşturuldu ve kaydedildi.")

            return exp.as_list()

        except Exception as e:
            print(f"LIME oluşturulurken hata oluştu: {e}. Model türü veya veri formatı ile ilgili sorun olabilir.")
            return []