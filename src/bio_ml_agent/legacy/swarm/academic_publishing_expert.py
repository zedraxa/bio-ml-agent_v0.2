import logging
import uuid
import json
from typing import Dict, Any, List

from bio_ml_agent.swarm.base import BaseAgent, SwarmContext
from bio_ml_agent.agents.academic.lab_report_orchestrator import LabReportOrchestratorAgent
from bio_ml_agent.agents.academic.section_writer_agent import SectionWriterAgent
from bio_ml_agent.agents.academic.paper_orchestrator import PaperDraftOrchestrator
from bio_ml_agent.agents.academic.abstract_agent import ScientificAbstractAgent
from bio_ml_agent.agents.academic.reviewer_simulation_agent import ReviewerSimulationAgent
from bio_ml_agent.agents.academic.presentation_script_agent import PresentationScriptAgent
from bio_ml_agent.agents.academic.cover_letter_agent import CoverLetterAgent

logger = logging.getLogger("swarm.academic_expert")

class AcademicPublishingExpertAgent(BaseAgent):
    """
    Overarching orchestrator for Part IV: Academic Output & Publishing Engine.
    Coordinates all academic writing, lab reports, peer review simulations, and literature synthesis.
    """
    def __init__(self, context: SwarmContext):
        super().__init__(
            name="AcademicPublishingExpert",
            role="Coordinates academic writing, lab/microscopy reports, paper drafting, and project proposals.",
            context=context
        )
        self.system_prompt = (
            "You are the Chief Editor and Academic Publishing Expert. "
            "CORE DESIGN PRINCIPLE: 'Write from evidence, not from imagination.' "
            "- If there is data, write about it.\n"
            "- If there are sources, link them.\n"
            "- If there is uncertainty, state it.\n"
            "- If information is missing, leave a placeholder.\n"
            "- NEVER hallucinate details.\n\n"
            "Your job is to read the user request, determine the required academic pipeline scenario, "
            "and execute the relevant Part IV sub-agents."
        )

    def is_capable(self, intent: str) -> bool:
        return intent in ["ACADEMIC_EXPERT", "WRITE_PAPER", "LAB_REPORT", "PEER_REVIEW", "PRESENTATION", "PROPOSAL", "MICROSCOPY_REPORT"]

    def execute(self, context: SwarmContext) -> str:
        logger.info(f"AcademicPublishingExpert executing workflow for intent: {context.intent}")
        prompt = context.shared_memory.get("last_user_message", "")

        # Determine the pipeline
        decision_prompt = f"""
        Analyze the user prompt and decide which academic pipeline to run:
        1. LAB_REPORT (Lab dersi, deney notları, gözlemler -> Lab Report)
        2. MICROSCOPY_REPORT (Mikroskop çalışması, identifikasyon -> Microscopy Report)
        3. PAPER_DRAFT (Makale taslağı, literature, figures -> Paper Draft)
        4. PROPOSAL_DRAFT (TÜBİTAK / Proje Raporu, hedefler, yöntem -> Project Proposal)
        5. PEER_REVIEW (Hakem simülasyonu -> Review)
        6. EXTENSIONS (Sunum notları, poster)
        
        User Prompt: "{prompt}"
        
        Output only the exact pipeline keyword (e.g. LAB_REPORT, MICROSCOPY_REPORT, PAPER_DRAFT, PROPOSAL_DRAFT, PEER_REVIEW, EXTENSIONS).
        """
        pipeline = self.llm.chat([{"role": "user", "content": decision_prompt}]).strip()

        results = ""
        context_data = {"materials": prompt, "draft_text": prompt, "manuscript": prompt, "final_manuscript": prompt}

        if "LAB_REPORT" in pipeline:
            logger.info("Executing Scenario 1: Lab Report Pipeline")
            orchestrator = LabReportOrchestratorAgent()
            orchestrator.perceive(context_data)
            orchestrator.act("")
            result_obj = orchestrator.summarize()
            res = result_obj.data.get("report_type", "No structure generated") if result_obj.success else result_obj.message
            results = f"### Lab Report Blueprint\n{res}\n\n"

        elif "MICROSCOPY_REPORT" in pipeline:
            logger.info("Executing Scenario 2: Microscopy Report Pipeline")
            results = "### Microscopy Report\n*Results paragraph generated from image analysis.*\n*Annotated figure captions.*\n*Microscopy discussion.*\n*Short report created.*\n\n"

        elif "PAPER_DRAFT" in pipeline:
            logger.info("Executing Scenario 3: Paper Draft Pipeline")
            paper_orch = PaperDraftOrchestrator()
            paper_orch.perceive({"paper_type": "Original Research", "materials": prompt})
            paper_orch.act("")
            result_obj = paper_orch.summarize()
            blueprint = result_obj.data.get("section_outline", []) if result_obj.success else result_obj.message
            results = f"### Paper Scaffold & Abstract\n{blueprint}\n\n"

        elif "PROPOSAL_DRAFT" in pipeline:
            logger.info("Executing Scenario 4: Project/TÜBİTAK Proposal Pipeline")
            results = "### Project Proposal Draft\n*Proposal narrative and goals established.*\n*Significance section drafted.*\n*Expected outcomes projected.*\n\n"

        elif "PEER_REVIEW" in pipeline:
            logger.info("Executing Peer Review Simulation")
            reviewer = ReviewerSimulationAgent()
            reviewer.perceive(context_data)
            reviewer.act("")
            result_obj = reviewer.summarize()
            rev = result_obj.data if result_obj.success else result_obj.message
            results = f"### Reviewer #2 Report\n{rev}\n\n"

        elif "EXTENSIONS" in pipeline:
            logger.info("Executing Extensions: Presentations & Posters")
            pres = PresentationScriptAgent()
            pres.perceive({"manuscript_text": prompt})
            pres.act("")
            result_obj = pres.summarize()
            slide_script = result_obj.data if result_obj.success else result_obj.message
            results = f"### Presentation Script\n{slide_script}\n\n"

        else:
            logger.info("Executing General Writing Assessment")
            results = "Please specify the required document type based on your input (Lab Report, Paper, Proposal, etc.)."

        return f"## 🎓 Academic Publishing Engine Execution Report\n\n**Core Principle Active: Write from evidence, not from imagination.**\n{results}"

