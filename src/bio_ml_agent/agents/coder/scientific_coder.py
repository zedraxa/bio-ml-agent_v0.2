import logging
import ast
import os
import uuid
from typing import List, Dict, Any, Optional
from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from bio_ml_agent.llm_backend import auto_create_backend

log = logging.getLogger("coder.scientific")

class ScientificPythonAgent(BaseSubAgent):
    """
    ScientificPythonAgent (B2): 'Bilimsel Kod Geliştiricisi' Uzmanı.
    Genel bir yazılımcı değil, numpy, pandas, scipy, scikit-learn, 
    matplotlib, statsmodels ve image processing kütüphanelerinde uzmandır.
    B1'den (Architect) gelen arayüz planlarını alır ve production-ready 
    analiz / pipeline kodları (veya Jupyter notebook'ları) üretir.
    """
    
    def __init__(self, model_name: str = "gemini-2.0-flash", **kwargs):
        super().__init__("ScientificCoder", model_name, **kwargs)
        # Using a higher tier model by default for rigorous code generation
        self.llm = auto_create_backend(model_name)
        self.architecture_spec: Dict[str, Any] = {}
        self.target_module: str = ""
        self.generated_code: str = ""

    def perceive(self, context: Dict[str, Any]) -> None:
        """Mimari bağlamını (B1 kaynaklı) ve uygulanacak hedef modülü algılar."""
        self.target_module = context.get("target_module", "core_pipeline")
        self.dataset_schema = context.get("dataset_schema", "No specific schema provided.")
        log.info(f"🔬 Scientific Coder analyzing architecture for module: {self.target_module}")

    def plan(self, goal: str) -> List[str]:
        return [
            f"Review interfaces and config requirements for module: {self.target_module}",
            "Select optimal scientific libraries (NumPy, SciPy, Scikit-Learn, etc.)",
            "Generate optimized, highly vectorized Python code.",
            "Include inline documentation, typing, and docstrings.",
            "Parse AST (Abstract Syntax Tree) to verify code validity."
        ]

    def act(self, step: str) -> Any:
        if not self.target_module:
            return "Error: No target module defined to code."
            
        log.info(f"💻 Coding step: {step}")
        
        # Sadece kod üretme adımındaysak LLM'i çağır
        if "Generate optimized" in step:
            prompt = f"""
            You are a World-Class Scientific Python Coder. 
            You specialize EXCLUSIVELY in data-science, bioinformatics, image processing, 
            and high-performance computing using: NumPy, Pandas, SciPy, Scikit-Learn, 
            Matplotlib, Statsmodels, CV2, and Skimage.
            
            Context Architecture:
            {self.architecture_spec}
            
            Dataset Schema / Meta:
            {self.dataset_schema}
            
            Your Task:
            Write the production-grade Python script for the module: '{self.target_module}'.
            Follow these constraints:
            1. Use highly vectorized NumPy/SciPy operations where possible.
            2. Fully type-hint all functions and write descriptive variable names.
            3. Include detailed docstrings (NumPy format) explaining assumptions and units.
            4. Domain-Aware Plotting: If generating plots, strongly prefer bio-specific visualizations
               (e.g. Volcano plots, Cell measurement distributions, Feature importance) with full labels.
            5. Do NOT write conversational text. ONLY return the raw Python code enclosed in ```python formatting.
            """
            
            messages = [{"role": "user", "content": prompt}]
            
            try:
                response_text = self.llm.chat(messages)
                self.generated_code = self._extract_code_block(response_text)
                
                filename = f"scripts/sci_{uuid.uuid4().hex[:4]}.py"
                os.makedirs("scripts", exist_ok=True)
                with open(filename, "w") as f:
                    f.write(self.generated_code)
                self.generated_artifacts = [filename]
            except Exception as e:
                log.error(f"Scientific Coder LLM failure: {e}")
                self.generated_code = "# Error generating scientific code."
                self.generated_artifacts = []
                
        return "Scientific analysis pipeline updated."

    def _extract_code_block(self, text: str) -> str:
        """LLM çıktısından sadece Python kodunu ayıklar."""
        if "```python" in text:
            code = text.split("```python")[1].split("```")[0].strip()
            return code
        elif "```" in text:
            # Fallback if language isn't specified
            code = text.split("```")[1].split("```")[0].strip()
            return code
        return text.strip()

    def verify(self, action_result: Any) -> bool:
        """Üretilen kodun sözdizimsel (Syntax) olarak geçerli olup olmadığını AST ile test eder."""
        if not self.generated_code or self.generated_code.startswith("# Error"):
            return False
            
        try:
            ast.parse(self.generated_code)
            log.info("✅ AST Verification Pass: Code is syntactically valid.")
            return True
        except SyntaxError as e:
            log.warning(f"❌ AST Verification Failed! Syntax error in generated code: {e}")
            return False

    def summarize(self) -> AgentResult:
        is_valid = self.verify("")
        conf = Confidence.HIGH if is_valid else Confidence.LOW
        msg = f"Scientific code generated for '{self.target_module}'." 
        if not is_valid:
            msg += " (WARNING: Code contains syntax errors!)"
            
        return self.create_result(
            success=is_valid,
            data={"target": self.target_module, "code": self.generated_code},
            confidence=conf,
            evidence=[Evidence(source="ast_parser", content_snippet="Python code structure is valid.")],
            message=f"Scientific code generation for {self.target_module} finished.",
            artifacts=getattr(self, "generated_artifacts", [])
        )
