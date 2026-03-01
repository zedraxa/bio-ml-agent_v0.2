"""ML Expert sub-agent for the Swarm Architecture."""
import logging
import re
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

from .base import BaseAgent, SwarmContext

class MLExpertAgent(BaseAgent):
    def __init__(self, context: SwarmContext):
        super().__init__(name="ML Expert", role="Machine Learning", context=context)
        self.system_prompt = (
            "Sen Bio-ML Swarm Topluluğunun 'Makine Öğrenimi Uzmanı'sın.\n"
            "Görevin: Veri setlerini okuyup modelleri eğitmek, karşılaştırmak ve sonuçları üretmektir.\n\n"
            "Araçların:\n"
            "1. Geleneksel ML: scikit-learn + utils/model_compare.py + utils/visualize.py\n"
            "2. Derin Öğrenme (CNN): from deep_learning import quick_train_cnn, compare_architectures\n"
            "   - ÖRN: quick_train_cnn('data/raw/brain_mri', preset='brain_mri', architecture='resnet18', epochs=2)\n"
            "   - DİKKAT: Path HER ZAMAN medikal görüntülerin (glioma vb.) olduğu alt klasör olmalıdır. Asla 'data/raw/' YAZMA.\n"
            "   - Preset'ler: chest_xray, brain_mri, skin_lesion, retinal_oct\n"
            "   - Mimariler: resnet18, resnet50, efficientnet_b0, densenet121, mobilenet_v2\n"
            "3. AutoML: from deep_learning import AutoMLSearch\n"
            "4. XAI (SHAP/LIME) - ZORUNLU ADIM: from xai_engine import XAIEngine\n"
            "   - Model eğitiminden sonra EN İYİ model için MUTLAKA SHAP veya LIME analizi yap.\n"
            "   - ÖRN: xai = XAIEngine(best_model, X_train, feature_names=cols)\n"
            "   - shap_dict = xai.generate_shap_summary(X_test, 'results/plots')\n"
            "   - LIME: xai.explain_instance_lime(X_test.iloc[0], 'results/plots')\n"
            "   - DİKKAT: Üretilen grafik yollarını ve en önemli özellik isimlerini (SHAP Feature Importance) print() ile ekrana yazdır ki Bioinfo uzmanı bunları okuyup yorumlayabilsin!\n\n"
            "ÇOK ÖNEMLİ: Kod yazarken her zaman <PYTHON> kod bölümü </PYTHON> taglarını kullanmak ZORUNDASIN. "
            "Markdown kod blokları (```python) ÇALIŞMAZ. Sadece <PYTHON> tagleri içindeki kodlar çalıştırılır.\n"
            "Görüntü sınıflandırma isteklerinde deep_learning modülünü kullan.\n"
            "Tablo veri isteklerinde scikit-learn pipeline kullan.\n"
            "ZORUNLU: Eğitim bittikten sonra mutlaka xai_engine üzerinden açıklanabilirlik sağla ve grafikleri üret.\n"
            "Cevabının sonunda her zaman sonuç özetini paylaş."
        )
        
    def get_system_prompt(self) -> str:
        return self.system_prompt
        
    def execute(self, task_prompt: str = "", error_history: str = "") -> str:
        """ML Expert LLM zincirini başlatır."""
        from llm_backend import auto_create_backend
        from agent import extract_tools, run_python
        from progress import Spinner
        
        backend = auto_create_backend(self.context.model)
        
        messages = [{"role": "system", "content": self.system_prompt}]
        
        if error_history:
            messages.append({"role": "system", "content": f"ÖNEMLİ HATA UYARISI: Önceki denemede hata alındı. Lütfen düzelt:\n{error_history}"})
            
        if task_prompt:
            messages.append({"role": "user", "content": task_prompt})
        elif self.context.history:
            messages.append(self.context.history[-1])
        
        logger.info("[ML Expert] Model eğitim ve değerlendirme görevine başlanıyor...")
        
        if "data_engineer_last_status" in self.context.shared_memory:
            messages.append({
                "role": "system", 
                "content": f"Bilgi: Veri Mühendisi işlemi bitirdi: {self.context.shared_memory['data_engineer_last_status']}"
            })

        max_steps = 10
        final_answer = ""
        
        for step in range(max_steps):
            with Spinner(f"🧠 ML Expert Düşünüyor (Adım {step+1}/{max_steps})"):
                response = backend.chat(messages)
            
            tools_to_run, outside = extract_tools(response)
            
            if not tools_to_run:
                py_m = re.search(r"<PYTHON>\s*(.*?)\s*</PYTHON>", response, re.DOTALL)
                if py_m:
                    tools_to_run = [("PYTHON", py_m.group(1))]

            messages.append({"role": "assistant", "content": response})
            
            if not tools_to_run:
                final_answer = response
                break
                
            all_outputs = []
            for tool, payload in tools_to_run:
                if tool == "PYTHON":
                    from pathlib import Path
                    py_cwd = Path(self.context.workspace)
                    py_cwd.mkdir(parents=True, exist_ok=True)
                    with Spinner("🐍 ML Expert Python Çalıştırıyor"):
                        out = run_python(payload, py_cwd, timeout_s=120)
                    
                    formatted_out = f"\n🛠️ PYTHON output:\n{out}\n"
                    all_outputs.append(formatted_out)
                    print(formatted_out)
                else:
                    all_outputs.append(f"[BLOCKED] ML Expert sadece PYTHON aracı kullanabilir.")
            
            messages.append({"role": "user", "content": "\n".join(all_outputs)})
        
        self.context.shared_memory["ml_expert_last_status"] = "Modeller eğitildi."
        return final_answer if final_answer else "ML Uzmanı döngüsü sona erdi."
