import logging
import ast
from typing import List, Dict, Any, Optional
from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from bio_ml_agent.llm_backend import auto_create_backend

log = logging.getLogger("coder.biomedical_ml")

class BiomedicalMLCodingAgent(BaseSubAgent):
    """
    BiomedicalMLCodingAgent (B4): 'Biyomedikal Yapay Zeka' Kodlayıcısı.
    Şu spesifik analizleri kurgular:
    - Tabular Biomedical ML & Imaging ML hedefleri
    - Classification, Regression, Survival Analysis
    - ML Explainability (SHAP, LIME vb.)
    - Model Evaluation & Error Analysis
    - Tekrar üretilebilir (Reproducibility) Jupyter Notebook'ları yazma
    """
    
    def __init__(self, model_name: str = "gemini-2.0-pro", **kwargs):
        super().__init__("BiomedicalMLCoder", model_name, **kwargs)
        self.llm = auto_create_backend(model_name)
        self.architecture_spec: Dict[str, Any] = {}
        self.target_module: str = ""
        self.generated_code: str = ""

    def perceive(self, context: Dict[str, Any]) -> None:
        """Mimari detaylarını ve ML modül hedefini alır."""
        self.architecture_spec = context.get("architecture_spec", {})
        self.target_module = context.get("target_module", "ml_pipeline")
        self.dataset_schema = context.get("dataset_schema", "No dataset schema provided.")
        log.info(f"🧠 Biomedical ML Coder analyzing target module: {self.target_module}")

    def plan(self, goal: str) -> List[str]:
        return [
            f"Analyze model structure and target metrics for: {self.target_module}",
            "Design the preprocessing and survival/classification targets",
            "Generate explainability logic (SHAP/LIME) for model interpretations",
            "Produce comprehensive evaluation and error analysis tools",
            "Package output into reproducible script or IPython Notebook format"
        ]

    def act(self, step: str) -> Any:
        if not self.target_module:
            return "Error: No ML target defined."
            
        log.info(f"🤖 Training/Coding step: {step}")
        
        if "Generate" in step or "Produce" in step or "Package" in step:
            prompt = f"""
            You are an Expert Biomedical Machine Learning Engineer.
            Write Python code targeting tabular biomarker ML, bio-imaging ML, 
            Survival models (Cox, Kaplan-Meier), or Classification/Regression models.
            You must strongly incorporate EXPLAINABILITY (SHAP, LIME) and Error Analysis.
            
            Architecture Specification:
            {self.architecture_spec}
            
            Dataset Schema:
            {self.dataset_schema}
            
            Target Module to Implement: '{self.target_module}'
            
            Rules:
            1. Write clear, commented ML code using scikit-learn, lifelines, xgboost, SHAP.
            2. Follow best practices for Biomedical reproducibility (random states cross-validation).
            3. Domain-Aware Plotting: ALWAYS generate appropriate plots (ROC/PR curves, Confusion Matrix, 
               Survival Kaplan-Meier, SHAP Summary Plots, Class distributions).
            4. Enforce strict scientific style: clear assumptions, parameter types, bounds, units.
            5. Return strictly the RAW PYTHON CODE surrounded by ```python markdown. No conversational filler.
            """
            
            messages = [{"role": "user", "content": prompt}]
            
            try:
                response_text = self.llm.chat(messages)
                self.generated_code = self._extract_code_block(response_text)
            except Exception as e:
                log.error(f"Biomedical ML Coder failure: {e}")
                self.generated_code = "# Error generating ML code."
                
        return f"Completed Biomedical ML coding step: {step}"

    def _extract_code_block(self, text: str) -> str:
        if "```python" in text:
            return text.split("```python")[1].split("```")[0].strip()
        elif "```" in text:
            return text.split("```")[1].split("```")[0].strip()
        return text.strip()

    def verify(self, action_result: Any) -> bool:
        if not self.generated_code or self.generated_code.startswith("# Error"):
            return False
        try:
            ast.parse(self.generated_code)
            return True
        except SyntaxError as e:
            log.warning(f"❌ Syntax Error in ML code: {e}")
            return False

    def summarize(self) -> AgentResult:
        is_valid = self.verify("")
        conf = Confidence.HIGH if is_valid else Confidence.LOW
        
        return self.create_result(
            success=is_valid,
            data={"module": self.target_module, "code": self.generated_code},
            confidence=conf,
            evidence=[Evidence(source="ast_parser", content_snippet="Biomedical ML syntax validated.")],
            message=f"Biomedical Machine Learning code generated for '{self.target_module}'."
        )
