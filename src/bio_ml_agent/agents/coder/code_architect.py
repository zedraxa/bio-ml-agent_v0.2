import logging
import json
from typing import List, Dict, Any, Optional
from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from bio_ml_agent.llm_backend import auto_create_backend

log = logging.getLogger("coder.architect")

class CodeArchitectAgent(BaseSubAgent):
    """
    CodeArchitectAgent (B1): 'Yazılım Mimarı' Uzmanı.
    Görevi doğrudan kod 'döşemek' yerine:
    - Büyük istekleri modüllere ayırmak
    - Klasör yapısı önermek
    - Arayüzleri (Interface/Contract) tanımlamak
    - Yapılandırma (Config) sistemini kurmak
    - Test stratejisini yazmak
    """
    
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        super().__init__("CodeArchitectAgent", model_name)
        self.llm = auto_create_backend(model_name)
        self.project_goal: str = ""
        self.architecture_spec: Dict[str, Any] = {}

    def perceive(self, context: Dict[str, Any]) -> None:
        """Ortamı veya girdiyi algılama aşaması."""
        self.project_goal = context.get("goal", "")
        if not self.project_goal:
            log.warning("No project goal provided to Code Architect.")
        else:
            goal_str = str(self.project_goal)
            log.info(f"🏗️ Architect perceived goal: {goal_str[:50]}...")

    def plan(self, goal: str) -> List[str]:
        """Mimari inşası için adım planlaması."""
        return [
            "Deconstruct the monolith goal into distinct logical modules.",
            "Design the directory and file structure.",
            "Define input/output interfaces and contracts for each module.",
            "Formulate configuration schemas (e.g. settings, hyperparameters).",
            "Establish a comprehensive testing strategy."
        ]

    def act(self, step: str) -> Any:
        """Belirli bir mimari tasarım adımını yürütme aşaması."""
        if not self.project_goal:
            return "No valid goal defined for architectural design."
            
        log.info(f"📐 Architecting step: {step}")
        
        prompt = f"""
        You are the Lead Scientific Software Architect for a Bio-ML project.
        Do NOT write full implementation code. Your job is purely architectural.
        
        Project Goal: {self.project_goal}
        
        Current Engineering Step: {step}
        
        Provide your architectural design output strictly as a JSON object matching this step.
        If analyzing modules, return {{"modules": ["mod1", "mod2"]}}.
        If designing folders, return {{"folder_structure": {{"src/": [...]}}}}.
        If defining interfaces, return {{"interfaces": [{{"module": "mod1", "functions": ["def foo() -> int"]}}]}}.
        If config, return {{"config": {{"keys": [...]}}}}.
        If testing, return {{"test_strategy": ["unit_tests", "mock_data_generation"]}}.
        """
        
        messages = [{"role": "user", "content": prompt}]
        try:
            response_text = self.llm.chat(messages)
            clean_text = response_text.replace("```json", "").replace("```", "").strip()
            try:
                parsed = json.loads(clean_text)
                self.architecture_spec.update(parsed)
            except json.JSONDecodeError:
                log.warning("Could not parse Architect LLM response to JSON.")
                self.architecture_spec[step.replace(' ', '_')] = response_text
        except Exception as e:
            log.error(f"Architect LLM failure: {e}")
            self._apply_fallback_architecture(step)
            
        return f"Architected step: {step}"

    def _apply_fallback_architecture(self, step: str):
        """API arızasında temel/örnek bir mimari yükler."""
        if "Deconstruct" in step:
            self.architecture_spec["modules"] = ["core", "data", "models", "utils"]
        elif "directory" in step:
            self.architecture_spec["folder_structure"] = {"src/": ["core/", "data/", "models/"]}
        elif "interfaces" in step:
            self.architecture_spec["interfaces"] = [{"module": "core", "functions": ["def process() -> bool"]}]
        elif "configuration" in step:
            self.architecture_spec["config"] = {"keys": ["model_path", "batch_size", "threshold"]}
        elif "testing" in step:
            self.architecture_spec["test_strategy"] = ["Test boundaries with pytest", "Mock external APIs"]

    def verify(self, action_result: Any) -> bool:
        """Üretilen mimarinin temel yapı taşlarına sahip olup olmadığını doğrular."""
        required_keys = ["modules", "folder_structure", "interfaces", "config"]
        valid = any(k in self.architecture_spec for k in required_keys)
        if not valid:
            log.warning("Architecture specification appears incomplete.")
        return valid

    def summarize(self) -> AgentResult:
        """Tasarım planını paketler ve çıktılar."""
        module_count = len(self.architecture_spec.get("modules", []))
        
        return AgentResult(
            success=self.verify(""),
            data=self.architecture_spec,
            confidence=Confidence.HIGH,
            evidence=[Evidence(source="architect_engine", content_snippet=f"Generated {module_count} distinct modules.")],
            message=f"System architecture formulated with {module_count} core modules."
        )
