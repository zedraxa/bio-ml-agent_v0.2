import logging
from typing import Dict, Any, List

from bio_ml_agent.kernel.message_bus import MessageBus
from bio_ml_agent.missions.base_mission import BaseMission

# Agent Imports
from bio_ml_agent.agents.biology.cell_biology_agent import CellBiologyAgent
from bio_ml_agent.agents.biology.wet_lab_protocol_agent import WetLabProtocolAgent
from bio_ml_agent.agents.coder.biomedical_ml_coder import BiomedicalMLCodingAgent
from bio_ml_agent.agents.coder.code_architect import CodeArchitectAgent
from bio_ml_agent.agents.coder.refactor_repair_agent import RefactorRepairAgent
# Simulated/Vision imports
# from bio_ml_agent.agents.vision.microscopy_perception import MicroscopyPerceptionAgent
# ... etc.

log = logging.getLogger("mission.integration")


class D1_LiteratureToExperimentWorkflow(BaseMission):
    """
    D1. Literature-to-Experiment Workflow
    Akış: Makaleleri bul -> Ortak bulguları çıkar -> Boşlukları bul -> 
          Deney öner -> Veri toplama ihtiyaçlarını yaz -> Analiz pipeline öner
    """
    def __init__(self, bus: MessageBus, topic: str):
        super().__init__("D1_LiteratureToExperiment", bus)
        self.topic = topic
        self.output: Dict[str, Any] = {}

    def setup(self):
        log.info(f"Setting up D1 Workflow for topic: {self.topic}")

    def execute(self) -> Dict[str, Any]:
        log.info("D1 Step 1: Searching for literature.")
        log.info("D1 Step 2: Extracting common findings and gaps.")
        log.info("D1 Step 3: Suggesting experiment.")
        log.info("D1 Step 4: Writing data collection needs and suggesting analysis pipeline.")
        
        self.output = {
            "topic": self.topic,
            "findings": ["Finding 1", "Finding 2"],
            "gaps": ["Unexplored mechanism X"],
            "suggested_experiment": "Perform CRISPR KO on mechanism X",
            "data_needs": "RNA-Seq and Confocal Microscopy",
            "analysis_pipeline": "Transcriptomics Differential Expression pipeline"
        }
        return self.output


class D2_MicroscopyToReportWorkflow(BaseMission):
    """
    D2. Microscopy-to-Report Workflow
    Akış: Görüntüleri yükle -> Kalite analizi -> Identification -> 
          Segmentation -> Ölçüm -> Annotation overlay -> Sonuç raporu
    """
    def __init__(self, bus: MessageBus, image_path: str):
        super().__init__("D2_MicroscopyToReport", bus)
        self.image_path = image_path
        self.report: Dict[str, Any] = {}

    def setup(self):
        log.info(f"Setting up D2 Workflow for image: {self.image_path}")

    def execute(self) -> Dict[str, Any]:
        log.info("D2 Step 1: Uploading images and QA (A1 Agent).")
        log.info("D2 Step 2: Identification and Segmentation (A2/A3/A4).")
        log.info("D2 Step 3: Morphometrics Measurement (A5).")
        log.info("D2 Step 4: Annotation Overlay.")
        
        # Call Biologist to write the final clinical report (C1/C2)
        log.info("D2 Step 5: Sending measurements to Biologist Agent for Final Report.")
        features = {"cell_count": 1500, "mitosis_rate": 0.05, "morphology": "irregular"}
        bio_agent = CellBiologyAgent()
        bio_agent.perceive({"features": features})
        bio_agent.act(bio_agent.plan("Write final biological insight")[0])
        
        self.report = {
            "image": self.image_path,
            "measurements": features,
            "biological_insight": bio_agent.summarize().data.get("biological_evaluation")
        }
        return self.report


class D3_DatasetToModelWorkflow(BaseMission):
    """
    D3. Dataset-to-Model Workflow
    Akış: Veri profille -> Uygun problem tipini seç -> Baseline üret -> 
          Evaluate et -> Explain et -> Report yaz
    """
    def __init__(self, bus: MessageBus, dataset_path: str):
        super().__init__("D3_DatasetToModel", bus)
        self.dataset_path = dataset_path
        self.output: Dict[str, Any] = {}

    def setup(self):
        log.info(f"Setting up D3 Workflow for dataset: {self.dataset_path}")

    def execute(self) -> Dict[str, Any]:
        log.info("D3 Step 1: Profiling data and selecting problem type (Architect B1).")
        architect = CodeArchitectAgent()
        architect.perceive({"task": f"Profile and plan ML for {self.dataset_path}"})
        arch_plan = architect.plan("Design ML pipeline")[0]
        architect.act(arch_plan)
        
        log.info("D3 Step 2: Generaing baseline ML code, evaluating, and explaining (Coder B4).")
        ml_coder = BiomedicalMLCodingAgent()
        ml_coder.perceive({"architecture": architect.summarize().data})
        for plan_step in ml_coder.plan("Generate Explainable ML Modeling Code"):
            ml_coder.act(plan_step)
            
        log.info("D3 Step 3: Writing final benchmark/report (Benchmark B6).")
        
        self.output = {
            "dataset": self.dataset_path,
            "target_model": "RandomForest / SHAP TreeExplainer",
            "code_generated": True
        }
        return self.output


class D4_CodeToRefactorWorkflow(BaseMission):
    """
    D4. Code-to-Refactor Workflow
    Akış: Repo tara -> Mimariyi özetle -> Sorunları bul -> 
          Patch planı yaz -> Test oluştur -> Değişiklik öner
    """
    def __init__(self, bus: MessageBus, repo_path: str):
        super().__init__("D4_CodeToRefactor", bus)
        self.repo_path = repo_path
        self.output: Dict[str, Any] = {}

    def setup(self):
        log.info(f"Setting up D4 Workflow for repository: {self.repo_path}")

    def execute(self) -> Dict[str, Any]:
        log.info("D4 Step 1: Scanning repo and summarizing architecture (B1).")
        log.info("D4 Step 2: Finding issues and writing patch plan (B5 RefactorRepair).")
        
        refactor = RefactorRepairAgent()
        refactor.perceive({"traceback": "Dummy Traceback: ModuleNotFoundError", "code": "import non_existent"})
        refactor.act(refactor.plan("Fix bugs")[0])
        
        log.info("D4 Step 3: Suggesting changes and generating tests.")
        
        self.output = {
            "repo": self.repo_path,
            "patched_code": refactor.summarize().data.get("patched_code")
        }
        return self.output


class D5_ProtocolToChecklistWorkflow(BaseMission):
    """
    D5. Protocol-to-Checklist Workflow
    Akış: Protokolü oku -> Reagent ve adımları çıkar -> Kritik noktaları işaretle -> 
          Zaman planı hazırla -> Hata risklerini listele
    """
    def __init__(self, bus: MessageBus, protocol_text: str):
        super().__init__("D5_ProtocolToChecklist", bus)
        self.protocol_text = protocol_text
        self.output: Dict[str, Any] = {}

    def setup(self):
        log.info("Setting up D5 Workflow for protocol text parsing.")

    def execute(self) -> Dict[str, Any]:
        log.info("D5 Step 1: Reading protocol and sending to C6 (WetLabProtocolAgent).")
        wetlab = WetLabProtocolAgent()
        wetlab.perceive({"methodology_text": self.protocol_text})
        
        log.info("D5 Step 2: Extracting reagents/steps/risks via C6.")
        wetlab.act(wetlab.plan("Extract wetlab variables")[0])
        
        self.output = {
            "protocol_source_length": len(self.protocol_text),
            "wetlab_checklist_and_risks": wetlab.summarize().data.get("wetlab_protocol")
        }
        return self.output
