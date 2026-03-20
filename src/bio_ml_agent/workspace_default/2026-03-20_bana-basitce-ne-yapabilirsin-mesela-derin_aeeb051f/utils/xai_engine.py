import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

class XAIEngine:
    """
    SHAP ve LIME kullanarak makine öğrenmesi modelleri için açıklamalar üreten bir sınıf.
    Model türünü algılayarak en verimli SHAP açıklayıcısını seçer.
    """
    def __init__(self, model, training_data, feature_names, class_names=['Benign', 'Malignant'], task_type="classification"):
        """
        Args:
            model (Pipeline): Eğitilmiş model pipeline'ı.
            training_data (pd.DataFrame): Modelin eğitildiği özellikler (X_train).
            feature_names (list): Özellik isimleri.
            class_names (list): Sınıf etiketleri.
            task_type (str): Görev tipi ('classification').
        """
        self.model = model
        self.training_data = training_data
        self.feature_names = list(feature_names)
        self.class_names = class_names
        self.task_type = task_type
        
        # LIME Explainer Kurulumu (değişiklik yok)
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.training_data.values,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode=self.task_type
        )
        
        # --- Akıllı SHAP Explainer Seçimi ---
        # Pipeline'dan ölçekleyiciyi ve asıl modeli çıkar
        scaler = self.model.named_steps.get('scaler')
        classifier = self.model.named_steps.get('model')

        background_data = shap.sample(self.training_data, 100)

        # Model tipini kontrol et ve uygun açıklayıcıyı seç
        if isinstance(classifier, LogisticRegression):
            print("Model LogisticRegression. Hızlı 'shap.LinearExplainer' kullanılıyor.")
            # LinearExplainer, modelin beklediği ölçeklenmiş veriyi gerektirir.
            transformed_background = scaler.transform(background_data)
            self.shap_explainer = shap.LinearExplainer(classifier, transformed_background)
        else:
            print(f"Model {type(classifier).__name__}. Yavaş 'shap.KernelExplainer' kullanılacak.")
            # HATA DÜZELTME: KernelExplainer için lambda fonksiyonu kullan
            predict_fn = lambda x: self.model.predict_proba(pd.DataFrame(x, columns=self.feature_names))
            self.shap_explainer = shap.KernelExplainer(predict_fn, background_data)
            
        print("XAIEngine (SHAP ve LIME) başlatıldı.")

    def generate_shap_summary(self, data_to_explain, output_dir="results/plots", max_display=15):
        """
        Verilen veri seti için bir SHAP özet grafiği (beeswarm) oluşturur ve kaydeder.
        """
        print("\n--- SHAP Özet Grafiği Oluşturuluyor ---")
        os.makedirs(output_dir, exist_ok=True)
        
        # Eğer explainer lineer ise, veriyi önce dönüştürmeliyiz
        if isinstance(self.shap_explainer, shap.explainers.Linear):
            scaler = self.model.named_steps.get('scaler')
            data_for_shap = scaler.transform(data_to_explain)
        else:
            data_for_shap = data_to_explain

        # SHAP değerlerini hesapla
        shap_values = self.shap_explainer.shap_values(data_for_shap)
        
        # Explainer'a göre SHAP değerlerinin formatı değişir.
        # KernelExplainer [class0, class1] listesi, LinearExplainer ise sadece class1 array'i döndürür.
        if isinstance(self.shap_explainer, shap.explainers.Kernel):
             shap_values_for_plot = shap_values[1] # Pozitif sınıf (Malignant)
        else: # Linear, Tree vb.
             shap_values_for_plot = shap_values

        plt.figure()
        # Önemli: summary_plot'a orijinal, okunabilir veriyi (data_to_explain) vermeliyiz.
        shap.summary_plot(shap_values_for_plot, data_to_explain, feature_names=self.feature_names, max_display=max_display, show=False)
        plt.title('SHAP Özet Grafiği (Malignant Sınıfı için)', fontsize=14)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, 'shap_summary_plot.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"SHAP özeti kaydedildi: {plot_path}")

    def explain_instance_lime(self, instance_to_explain, output_dir="results/plots", num_features=10):
        """
        Tek bir veri örneği için LIME açıklaması oluşturur ve HTML raporu olarak kaydeder.
        """
        print("\n--- Tekil Örnek için LIME Açıklaması Oluşturuluyor ---")
        os.makedirs(output_dir, exist_ok=True)

        explanation = self.lime_explainer.explain_instance(
            instance_to_explain.values,
            self.model.predict_proba,
            num_features=num_features
        )
        
        report_path = os.path.join(output_dir, 'lime_instance_explanation.html')
        explanation.save_to_file(report_path)
        print(f"LIME raporu kaydedildi: {report_path}")
