import os
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class XAIEngine:
    """Tıbbi yapay zeka modelleri için Açıklanabilir Yapay Zeka (XAI) Motoru.
    
    SHAP (SHapley Additive exPlanations) ve LIME (Local Interpretable 
    Model-agnostic Explanations) metodlarını kullanarak modellerin
    kararlarını şeffaflaştırır.
    """
    
    def __init__(self, model, X_reference, feature_names=None, class_names=None, task_type="classification"):
        """
        Args:
            model: Eğitilmiş makine öğrenmesi modeli (örn: RandomForestClassifier)
            X_reference: Arka plan/referans veri seti (SHAP ve LIME için)
            feature_names: Özellik isimleri listesi
            class_names: Sınıf isimleri listesi
            task_type: "classification" veya "regression"
        """
        self.model = model
        
        # NumPy dizisini Pandas DataFrame'e dönüştür (SHAP plotları için daha iyi)
        if isinstance(X_reference, np.ndarray):
            self.feature_names = feature_names if feature_names is not None else [f"Feature_{i}" for i in range(X_reference.shape[1])]
            self.X_reference = pd.DataFrame(X_reference, columns=self.feature_names)
        else:
            self.X_reference = X_reference
            self.feature_names = feature_names if feature_names is not None else list(X_reference.columns)
            
        self.class_names = class_names if class_names is not None else ["Sinif_0", "Sinif_1"]
        self.task_type = task_type
        
        # Lazy imports ve Explainer kurulumları
        self.shap_explainer = None
        self.lime_explainer = None
        
        logger.info(f"XAI Engine başlatıldı. Görev tipi: {task_type}, Özellik sayısı: {len(self.feature_names)}")

    def _get_shap_explainer(self):
        """SHAP Explainer nesnesini tembel (lazy) olarak yükler ve döndürür."""
        if self.shap_explainer is None:
            import shap
            
            # Ağaç tabanlı modeller için TreeExplainer (daha hızlı)
            tree_models = ("RandomForest", "GradientBoosting", "DecisionTree", "ExtraTrees", "XGB", "LGBM", "CatBoost")
            model_type = type(self.model).__name__
            
            if any(name in model_type for name in tree_models):
                self.shap_explainer = shap.TreeExplainer(self.model)
                logger.debug("SHAP TreeExplainer başlatıldı.")
            else:
                # Diğer (Linear, SVM, NN vb.) modeller için KernelExplainer (yavaş ama genel)
                # KernelExplainer çok yavaş olduğu için referans verisi k-means ile küçültülür
                background = shap.kmeans(self.X_reference, min(50, len(self.X_reference)))
                predict_fn = self.model.predict_proba if self.task_type == "classification" and hasattr(self.model, "predict_proba") else self.model.predict
                self.shap_explainer = shap.KernelExplainer(predict_fn, background)
                logger.debug("SHAP KernelExplainer başlatıldı.")
                
        return self.shap_explainer

    def _get_lime_explainer(self):
        """LIME Explainer nesnesini tembel olarak yükler ve döndürür."""
        if self.lime_explainer is None:
            from lime.lime_tabular import LimeTabularExplainer
            
            mode = "classification" if self.task_type == "classification" else "regression"
            
            self.lime_explainer = LimeTabularExplainer(
                self.X_reference.values,
                feature_names=self.feature_names,
                class_names=self.class_names,
                mode=mode,
                discretize_continuous=True
            )
            logger.debug("LIME Tabular Explainer başlatıldı.")
            
        return self.lime_explainer

    def generate_shap_summary(self, X_sample, output_dir, max_display=15):
        """Verilen örneklem için SHAP Summary Plot üretir ve kaydeder.
        
        Bu grafik, hangi özelliklerin modelin kararlarında en çok etkili olduğunu 
        ve bu etkilerin yönünü (pozitif/negatif) toplu olarak gösterir.
        """
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        # NumPy dizisini DataFrame yap
        if isinstance(X_sample, np.ndarray):
            X_sample = pd.DataFrame(X_sample, columns=self.feature_names)
            
        # SHAP değerlerini hesapla
        explainer = self._get_shap_explainer()
        
        try:
            shap_values = explainer.shap_values(X_sample)
            
            # Sınıflandırma algoritmaları bazen liste döner (her sınıf için ayrı SHAP)
            # Binary sınıflandırma için genellikle 1. indeks pozitif sınıfı temsil eder
            if isinstance(shap_values, list) and len(shap_values) >= 2:
                plot_shap_values = shap_values[1]
                title = f"SHAP Değerleri ({self.class_names[1]} Sınıfı Üzerindeki Etki)"
            else:
                plot_shap_values = shap_values
                title = "Özelliklerin Önemi (SHAP Summary)"
            
            # Grafiği oluştur ve kaydet
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                plot_shap_values, 
                X_sample, 
                max_display=max_display, 
                show=False,
                plot_type="dot"
            )
            plt.title(title, pad=20, fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            save_path = out_path / "shap_summary_plot.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Bar plot (global önem sırası)
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                plot_shap_values, 
                X_sample, 
                max_display=max_display, 
                show=False,
                plot_type="bar"
            )
            plt.title("Genel Özellik Önemi Sıralaması", pad=20, fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            bar_save_path = out_path / "shap_feature_importance.png"
            plt.savefig(bar_save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"SHAP grafikleri kaydedildi: {output_dir}")
            return {
                "summary_plot": str(save_path),
                "feature_importance": str(bar_save_path)
            }
            
        except Exception as e:
            logger.error(f"SHAP grafik oluşturma hatası: {e}")
            return {"error": str(e)}

    def explain_instance_lime(self, instance, output_dir, file_name="lime_explanation.html", num_features=10):
        """Tek bir tahminin LIME raporunu HTML olarak kaydeder.
        
        Spesifik bir veri noktasının (örneğin bir hastanın verileri)
        neden o şekilde sınırlandırıldığını açıklar.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        explainer = self._get_lime_explainer()
        
        # NumPy array'e çevir
        if isinstance(instance, pd.Series):
            instance_arr = instance.values
        elif isinstance(instance, pd.DataFrame):
            instance_arr = instance.iloc[0].values
        else:
            instance_arr = np.array(instance).flatten()
            
        # Predict fonksiyonunu hazırla
        predict_fn = self.model.predict_proba if self.task_type == "classification" and hasattr(self.model, "predict_proba") else self.model.predict
        
        try:
            # Açıklamayı oluştur
            exp = explainer.explain_instance(
                instance_arr, 
                predict_fn, 
                num_features=num_features
            )
            
            # HTML olarak kaydet
            save_path = out_path / file_name
            exp.save_to_file(str(save_path))
            
            logger.info(f"LIME raporu kaydedildi: {save_path}")
            
            # Rapor bilgisini JSON olarak da dön (API vb. için faydalı)
            report_dict = {
                "prediction": exp.predict_proba.tolist() if hasattr(exp, "predict_proba") else [float(exp.predicted_value)],
                "html_path": str(save_path),
                "explanations": exp.as_list()
            }
            
            # JSON raporu da logla
            with open(out_path / f"{file_name.replace('.html', '.json')}", 'w') as f:
                json.dump(report_dict, f, indent=2, ensure_ascii=False)
                
            return report_dict
            
        except Exception as e:
            logger.error(f"LIME açıklama oluşturma hatası: {e}")
            return {"error": str(e)}
