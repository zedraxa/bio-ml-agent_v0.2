import logging
import ast
import os
import uuid
from typing import List, Dict, Any, Optional
from bio_ml_agent.core.agent_base import BaseSubAgent, AgentResult, Confidence, Evidence
from bio_ml_agent.llm_backend import auto_create_backend

log = logging.getLogger("coder.bioinformatics")

class BioinformaticsPythonAgent(BaseSubAgent):
    """
    BioinformaticsPythonAgent (B3): 'Omics & Sekans Kod Uzmanı'.
    Moleküler Biyoloji ve Genetik (MBG) alanındaki veri mühendisliği için özelleşmiştir:
    - FASTA/FASTQ işleme, okuma (sequence parsing)
    - Alignment wrappers (BWA, Bowtie vb. etrafında script)
    - Motif search
    - BLAST workflow tutkallama kodları (glue code)
    - Biopython bazlı pipeline'lar
    - Variant filtering (VCF) ve expression matrix işleme
    """
    
    def __init__(self, model_name: str = "gemini-2.5-pro", **kwargs):
        super().__init__("BioinformaticsPythonAgent", model_name, **kwargs)
        self.llm = auto_create_backend(model_name)
        self.architecture_spec: Dict[str, Any] = {}
        self.target_module: str = ""
        self.generated_code: str = ""

    def perceive(self, context: Dict[str, Any]) -> None:
        """B1'den gelen mimariyi ve analiz edilecek modülü (örn: fasta_processor) kurgular."""
        self.architecture_spec = context.get("architecture_spec", {})
        self.target_module = context.get("target_module", "omics_pipeline")
        log.info(f"🧬 Bioinformatics Coder tracking module targeted for: {self.target_module}")

    def plan(self, goal: str) -> List[str]:
        return [
            f"Cross-reference Biopython classes needed for {self.target_module}",
            "Design highly memory-efficient generators for large FASTQ/VCF files",
            "Generate optimized OMICS pipeline scripting",
            "Format the output strictly as Python AST-parsable code"
        ]

    def act(self, step: str) -> Any:
        if not self.target_module:
            return "Error: No target module defined for Omics programming."
            
        log.info(f"💾 Constructing genomic code step: {step}")
        
        if "Generate" in step or "Design" in step:
            prompt = f"""
            You are a Bioinformatics/Genomics Python Coder.
            You must write highly efficient, generator-based codes for: FASTA/FASTQ parsing,
            sequence alignment wrapper generation, variant filtering (VCF), expression matrix handling,
            and Biopython pipelines.
            
            Current Architecture Context:
            {self.architecture_spec}
            
            Task:
            Write the full production-grade Python code for the module: '{self.target_module}'.
            Constraints:
            1. Use 'Yield' heavily for large FASTQ parsing to prevent RAM overflow.
            2. Follow Biopython standards where applicable.
            3. Type-hint precisely (e.g., Sequence objects, Generators).
            4. Include genomic docstrings.
            5. Return ONLY RAW PYTHON CODE inside ```python markdown brackets. No conversational text.
            """
            
            messages = [{"role": "user", "content": prompt}]
            
            try:
                response_text = self.llm.chat(messages)
                self.generated_code = self._extract_code_block(response_text)
                
                filename = f"scripts/{self.target_module}_{uuid.uuid4().hex[:4]}.py"
                os.makedirs("scripts", exist_ok=True)
                with open(filename, "w") as f:
                    f.write(self.generated_code)
                self.generated_artifacts = [filename]
            except Exception as e:
                log.error(f"Bioinformatics Coder LLM failure: {e}")
                self.generated_code = "# Error generating bioinformatics code."
                self.generated_artifacts = []
                
        return f"Completed Omics integration for step: {step}"

    def _extract_code_block(self, text: str) -> str:
        """Markdown içerisinden Python betiğini ayıklar."""
        if "```python" in text:
            return text.split("```python")[1].split("```")[0].strip()
        elif "```" in text:
            return text.split("```")[1].split("```")[0].strip()
        return text.strip()

    def verify(self, action_result: Any) -> bool:
        """Kodu AST ağacına derleyerek sözdizimsel doğrulaması yapar."""
        if not self.generated_code or self.generated_code.startswith("# Error"):
            return False
            
        try:
            ast.parse(self.generated_code)
            log.info("✅ AST Pass: Bioinformatics code is structurally sound.")
            return True
        except SyntaxError as e:
            log.warning(f"❌ AST Syntax Error in genomics code: {e}")
            return False

    def summarize(self) -> AgentResult:
        is_valid = self.verify("")
        conf = Confidence.HIGH if is_valid else Confidence.LOW
        
        return self.create_result(
            success=is_valid,
            data={"module": self.target_module, "code": self.generated_code},
            confidence=conf,
            evidence=[Evidence(source="ast_parser", content_snippet="Genomics pipeline validated.")],
            message=f"Bioinformatics module '{self.target_module}' code generation finished.",
            artifacts=getattr(self, "generated_artifacts", [])
        )
