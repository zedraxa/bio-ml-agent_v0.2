import shap
import pandas as pd
import matplotlib.pyplot as plt
import os

class XAIEngine:
    """
    Handles model explainability using SHAP.
    """
    def __init__(self, model, training_data, feature_names, task_type="classification"):
        self.model = model
        self.training_data = training_data
        self.feature_names = feature_names
        self.task_type = task_type
        
        # SHAP explainer'ı pipeline'ın model adımı için oluştur
        # Scaler'dan geçmiş veri üzerinde açıklama yapmak daha doğrudur.
        self.preprocessor = self.model.named_steps['scaler']
        self.predictor = self.model.named_steps['model']
        
        # Veriyi öncelikle ölçeklendir
        scaled_training_data = self.preprocessor.transform(self.training_data)
        
        # SHAP için maskeleme (masker) oluştur
        self.masker = shap.maskers.Independent(scaled_training_data, max_samples=100)
        
        # Explainer'ı oluştur
        self.explainer = shap.Explainer(self.predictor, self.masker)
        print("XAIEngine başarıyla başlatıldı.")

    def generate_shap_summary(self, data_to_explain, output_dir="results/plots", max_display=15):
        """
        Generates and saves a SHAP summary plot (beeswarm).
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        print("SHAP değerleri hesaplanıyor... Bu işlem biraz sürebilir.")
        
        # Açıklanacak veriyi de ölçeklendir
        scaled_data_to_explain = self.preprocessor.transform(data_to_explain)
        
        # SHAP değerlerini hesapla
        shap_values = self.explainer(scaled_data_to_explain)
        
        # shap_values objesinin feature_names'i kullanmasını sağla
        shap_values.feature_names = self.feature_names
        
        # Beeswarm plot
        plt.figure()
        shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
        plt.title('SHAP Feature Importance Summary (Beeswarm)')
        plt.tight_layout()
        save_path = os.path.join(output_dir, "shap_summary_beeswarm.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"SHAP summary plot (beeswarm) kaydedildi: {save_path}")

        # Bar plot
        plt.figure()
        shap.plots.bar(shap_values, max_display=max_display, show=False)
        plt.title('SHAP Global Feature Importance (Bar)')
        plt.tight_layout()
        save_path = os.path.join(output_dir, "shap_summary_bar.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"SHAP summary plot (bar) kaydedildi: {save_path}")