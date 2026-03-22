import logging
from typing import List, Dict, Any, Optional
from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from bio_ml_agent.llm_backend import auto_create_backend

log = logging.getLogger("coder.refactor_repair")

class RefactorRepairAgent(BaseSubAgent):
    """
    RefactorRepairAgent (B5): 'Kod Onarım ve İyileştirme' Uzmanı.
    Bu ajan yeni kod üretmek yerine:
    - Mevcut kodu okur ve iyileştirir (Refactoring)
    - Anti-pattern tespiti yapar
    - Bug şüphesi listesi (Bug suspicion) çıkarır
    - Eksik/kırık importları, yolları ve kullanılmayan fonksiyonları onarır
    - Otomatik test dosyaları (pytest) önerir/yazar
    """
    
    def __init__(self, model_name: str = "gemini-2.5-pro"):
        super().__init__("RefactorRepairAgent", model_name)
        # Using a model capable of deep reasoning for bug hunting
        self.llm = auto_create_backend(model_name)
        self.target_code: str = ""
        self.analysis_report: Dict[str, Any] = {}
        self.patched_code: str = ""

    def perceive(self, context: Dict[str, Any]) -> None:
        """Denetlenecek mevcut kodu ve varsa hata mesajlarını algılar."""
        self.target_code = context.get("code", "")
        self.traceback_err = context.get("traceback", "")
        if not self.target_code:
            log.warning("No target code provided for Refactor/Repair Agent.")
        else:
            log.info(f"🛠️ Refactor agent ingested {len(self.target_code)} characters of code.")

    def plan(self, goal: str) -> List[str]:
        return [
            "Analyze codebase for structural integrity and biological ML anti-patterns.",
            "Identify broken imports, dead functions, or risky path references.",
            "Formulate a 'Bug Suspicion Profile' mapping potential runtime failures.",
            "Generate Test Cases to cover the risky behaviors.",
            "Synthesize and apply the final patched, refactored code block."
        ]

    def act(self, step: str) -> Any:
        if not self.target_code:
            return "Error: Empty code block."
            
        log.info(f"🔍 Repairing step: {step}")
        
        prompt = f"""
        You are a Staff-Level Software Engineer specialized in Bio-ML Code Audits (Refactoring & Repair).
        
        Original Code:
        ```python
        {self.target_code}
        ```
        
        Reported Traceback/Error (if any):
        {self.traceback_err}
        
        Current Phase: {step}
        
        If analyzing anti-patterns or bugs, return a strict JSON dictionary:
        {{"anti_patterns": ["...", "..."], "bug_suspicions": ["...", "..."], "broken_imports": ["..."]}}
        
        If generating a patch/refactor, return ONLY the full raw patched python code enclosed in ```python markdown.
        """
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            if "analyze" in step.lower() or "identify" in step.lower() or "suspicion" in step.lower():
                response_text = self.llm.chat(messages)
                import json
                clean_text = response_text.replace("```json", "").replace("```", "").strip()
                try:
                    self.analysis_report.update(json.loads(clean_text))
                except Exception:
                    self.analysis_report[step.replace(' ', '_')] = response_text
            else:
                response_text = self.llm.chat(messages)
                self.patched_code = self._extract_code_block(response_text)
        except Exception as e:
            log.error(f"Refactor Agent LLM failure: {e}")
            
        return f"Refactoring step {step} executed."

    def _extract_code_block(self, text: str) -> str:
        if "```python" in text:
            return text.split("```python")[1].split("```")[0].strip()
        elif "```" in text:
            return text.split("```")[1].split("```")[0].strip()
        return text.strip()

    def verify(self, action_result: Any) -> bool:
        """Yamalanmış (patched) kodun syntax error verip vermediğini kontrol eder."""
        if not self.patched_code:
            return False
            
        import ast
        try:
            ast.parse(self.patched_code)
            log.info("✅ AST Pass: Patched code is syntactically sound.")
            return True
        except SyntaxError as e:
            log.warning(f"❌ Syntax Error injected during refactor: {e}")
            return False

    def summarize(self) -> AgentResult:
        is_valid = self.verify("")
        
        return self.create_result(
            success=is_valid,
            data={
                "audit_report": self.analysis_report,
                "patched_code": self.patched_code
            },
            confidence=Confidence.HIGH if is_valid else Confidence.LOW,
            evidence=[Evidence(source="refactoring_engine", content_snippet="Audit complete.")],
            message="Code refactored and auto-repaired successfully." if is_valid else "Refactoring induced syntax problems."
        )
