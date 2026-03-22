import os
import sys
import logging
from pathlib import Path

# Add src to sys.path
src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from bio_ml_agent.services.agent_service import AgentService
from bio_ml_agent.db.session import SessionLocal, engine, Base
from bio_ml_agent.db.models import MissionDB, MissionStepDB, ArtifactDB, ProjectDB

def init_db():
    Base.metadata.create_all(bind=engine)
    print("✅ Database Initialized.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_mission_packs")

def verify_orchestration():
    print("\n" + "="*60)
    print("🚀 MISSION PACK ORCHESTRATION VERIFICATION")
    print("="*60)

    # 1. Initialize Service
    service = AgentService()
    
    # 2. Run repo_review_pack
    print("\n[SCENARIO 1] Repository Review Pack")
    try:
        mid_repo = service.run_mission_pack("repo_review_pack", "Audit the core orchestrator for security smells.")
        print(f"✅ Mission Started: {mid_repo}")
        
        # Verify persistence
        with SessionLocal() as db:
            mission = db.query(MissionDB).filter(MissionDB.mission_id == mid_repo).first()
            steps = db.query(MissionStepDB).filter(MissionStepDB.mission_id == mid_repo).all()
            artifacts = db.query(ArtifactDB).filter(ArtifactDB.mission_id == mid_repo).all()
            
            print(f"  - DB Status: {mission.status.value if mission else 'NOT FOUND'}")
            print(f"  - Steps Logged: {len(steps)}")
            for s in steps:
                print(f"    - [{s.action_type}] {s.agent_role}: {s.content[:50]}...")
            print(f"  - Artifacts Produced: {len(artifacts)}")
    except Exception as e:
        print(f"❌ repo_review_pack failed: {e}")

    # 3. Run microscopy_pack
    print("\n[SCENARIO 2] Microscopy Pack")
    try:
        mid_micro = service.run_mission_pack("microscopy_pack", "Process HeLa cell slides from /data/images.")
        print(f"✅ Mission Started: {mid_micro}")
        
        with SessionLocal() as db:
            mission = db.query(MissionDB).filter(MissionDB.mission_id == mid_micro).first()
            steps = db.query(MissionStepDB).filter(MissionStepDB.mission_id == mid_micro).all()
            print(f"  - DB Status: {mission.status.value if mission else 'NOT FOUND'}")
            print(f"  - Steps Logged: {len(steps)}")
    except Exception as e:
        print(f"❌ microscopy_pack failed: {e}")

    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    init_db()
    verify_orchestration()
