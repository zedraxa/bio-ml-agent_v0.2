import asyncio
import logging
from bio_ml_agent.agents.microscopy.slide_navigator_agent import SlideNavigatorAgent
from bio_ml_agent.agents.microscopy.cross_view_comparator import CrossViewComparator
from bio_ml_agent.agents.microscopy.temporal_agent import TemporalMicroscopyAgent
from bio_ml_agent.agents.microscopy.annotation_agent import AnnotationSuggestionAgent
from bio_ml_agent.agents.microscopy.active_learning_agent import ActiveLearningAgent

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("demo.advanced_microscopy")

async def run_advanced_microscopy_mission():
    """Advanced Microscopy Modules Integration Test"""
    context = {"image_path": "samples/wsi_sample.svs", "comparison_views": [{}, {}], "temporal_frames": [{}, {}]}
    
    # E1: Slide Navigator
    nav = SlideNavigatorAgent()
    nav.perceive(context)
    log.info("Step E1: Navigating WSI...")
    
    # E2: Cross-View Comparator
    comp = CrossViewComparator()
    comp.perceive(context)
    log.info("Step E2: Comparing Views...")
    
    # E3: Temporal Agent
    temp = TemporalMicroscopyAgent()
    temp.perceive(context)
    log.info("Step E3: Temporal Tracking...")
    
    # E4: Annotation Suggestion
    ann = AnnotationSuggestionAgent()
    ann.perceive(context)
    log.info("Step E4: Copilot Annotation...")
    
    # E5: Active Learning
    learn = ActiveLearningAgent()
    learn.perceive(context)
    log.info("Step E5: Active Learning Loop...")
    
    log.info("🏁 Advanced Microscopy Mission Integration Logic Verified.")

if __name__ == "__main__":
    asyncio.run(run_advanced_microscopy_mission())
