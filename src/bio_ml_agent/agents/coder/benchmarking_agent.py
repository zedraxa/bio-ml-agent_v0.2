import logging
import ast
from typing import List, Dict, Any, Optional
from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from bio_ml_agent.llm_backend import auto_create_backend

log = logging.getLogger("coder.benchmarking")

class BenchmarkingAgent(BaseSubAgent):
    """
    BenchmarkingAgent (B6): 'Kod Kalitesi ve Performans Test' Uzmanı.
    Diğer ajanların (B2, B3, B4) ürettiği kodların kalitesini "yazdım-bitti" 
    mantığıyla değil, "iyi yazıldı mı?" süzgecinden geçirir:
    - Performans / Zaman Karmaşıklığı (Runtime)
    - RAM / Bellek Yönetimi (Memory constraints, generator kullanımı vb.)
    - Okunabilirlik (Readability, PEP8)
    - Test Kapsamı (Test coverage)
    - Yeniden Üretilebilirlik (Reproducibility)
    """

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        super().__init__("BenchmarkingAgent", model_name)
        # Gemin 2.5 Flash is highly capable of reading fast code structure
        self.llm = auto_create_backend(model_name)
        self.code_to_review: str = ""
        self.benchmark_report: Dict[str, Any] = {}

    def perceive(self, context: Dict[str, Any]) -> None:
        """Denetlenecek kaynak kodu bağlamdan çeker."""
        self.code_to_review = context.get("code", "")
        if not self.code_to_review:
            log.warning("No code provided for Benchmarking.")
        else:
            log.info(f"⏱️ Benchmarker received code block ({len(self.code_to_review)} chars) for auditing.")

    def plan(self, goal: str) -> List[str]:
        return [
            "Analyze Big-O runtime complexity of loops and DataFrame operations.",
            "Inspect memory allocations (lists vs generators, heavy copying).",
            "Audit strictly for scientific reproducibility (random seeds, explicit versions).",
            "Evaluate testability (pure functions vs side effects).",
            "Generate a 'Benchmark Scorecard' as JSON format."
        ]

    def act(self, step: str) -> Any:
        if not self.code_to_review:
            return "Error: No code to benchmark."

        log.info(f"📊 Benchmarking step: {step}")

        prompt = f"""
        You are the Quality Control & Performance Architect (Benchmarking Agent) for a Scientific CI/CD pipeline.
        
        Code to Benchmark:
        ```python
        {self.code_to_review}
        ```
        
        Current Audit Phase: {step}
        
        Your ONLY job is to return a strict JSON dictionary evaluating the code against these factors:
        {{
            "performance_score": int (0-100),
            "memory_score": int (0-100),
            "readability_score": int (0-100),
            "reproducibility_score": int (0-100),
            "critical_bottlenecks": ["..."],
            "improvement_suggestions": ["..."]
        }}
        Do NOT wrap the response in conversational text. Just the raw JSON (or JSON bounded by ```json markdown).
        """

        messages = [{"role": "user", "content": prompt}]

        try:
            response_text = self.llm.chat(messages)
            import json
            clean_text = response_text.replace("```json", "").replace("```", "").strip()
            self.benchmark_report = json.loads(clean_text)
        except Exception as e:
            log.warning(f"Benchmarking analysis parsing fell back to text. Error: {e}")
            self.benchmark_report[step.replace(' ', '_')] = response_text

        return f"Completed Benchmarking step: {step}"

    def verify(self, action_result: Any) -> bool:
        """Hedef metriklerin (score'ların) başarıyla üretildiğini denetler."""
        required = ["performance_score", "memory_score", "readability_score", "reproducibility_score"]
        valid = any(k in self.benchmark_report for k in required)
        if not valid:
            log.error("Benchmark report is structurally invalid or incomplete.")
        return valid

    def summarize(self) -> AgentResult:
        is_valid = self.verify("")

        # Calculate overall quality if scores exist
        scores = [v for k, v in self.benchmark_report.items() if k.endswith("_score") and isinstance(v, (int, float))]
        overall_score = sum(scores) / len(scores) if scores else 0

        msg = f"Code benchmarked. Overall Score: {overall_score:.1f}/100"
        conf = Confidence.HIGH if overall_score >= 80 else Confidence.LOW

        return AgentResult(
            success=is_valid,
            data={"scorecard": self.benchmark_report, "overall_score": overall_score},
            confidence=conf,
            evidence=[Evidence(source="benchmark_engine", content_snippet=f"Average rating {overall_score:.1f}")],
            message=msg
        )
