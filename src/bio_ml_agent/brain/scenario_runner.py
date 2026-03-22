import logging
import time
from typing import List, Dict, Any, Optional
from .scenario_replay import ScenarioReplaySystem, RegressionReport
from .models import MissionPlan
from .mission_brain import MissionBrain

logger = logging.getLogger("bio_ml_agent.brain.testing")

class ScenarioTestRunner:
    """
    Part VI: Scenario Test Runner.
    
    Coordinates the execution of scenario tests and generates 
    regression reports against golden snapshots.
    """

    def __init__(self, brain: MissionBrain, replay_system: ScenarioReplaySystem):
        self.brain = brain
        self.replay = replay_system

    def run_test(self, snapshot_id: str, prompt: str) -> RegressionReport:
        """Re-runs a mission prompt and compares the result to a golden snapshot."""
        logger.info(f"[ScenarioRunner] Starting regression test for: {snapshot_id}")
        
        # 1. Load Golden Original
        golden = self.replay.load_snapshot(snapshot_id)
        if not golden:
            raise ValueError(f"Snapshot {snapshot_id} not found.")

        # 2. Execute Fresh Mission
        start_time = time.time()
        new_plan = self.brain.decompose(prompt)
        # Mocking execution here for demo purposes; real system would call .execute()
        duration = time.time() - start_time
        
        logger.info(f"[ScenarioRunner] Fresh execution finished in {duration:.2f}s")

        # 3. Compare and Report
        # Note: In a real run, we'd need the resulting ArtifactGraph as well
        from .artifact_graph import ArtifactGraph
        empty_graph = ArtifactGraph(project_id="test") # Placeholder
        
        report = self.replay.compare(golden, new_plan, empty_graph)
        
        if report.is_regression:
            logger.error(f"[ScenarioRunner] REGRESSION DETECTED: {len(report.issues)} issues.")
        else:
            logger.info(f"[ScenarioRunner] Test PASSED.")
            
        return report

    def batch_run(self, tests: List[Dict[str, str]]) -> List[RegressionReport]:
        """Runs a suite of tests."""
        reports = []
        for t in tests:
            report = self.run_test(t["snapshot_id"], t["prompt"])
            reports.append(report)
        return reports
