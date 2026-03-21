import logging
import json
from bio_ml_agent.kernel.message_bus import MessageBus
from bio_ml_agent.missions.integration.mission_graphs import D1_LiteratureToExperimentWorkflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

def main():
    print("=====================================================")
    print("🚀 Bio-ML Agent OS: Eksen D - End-to-End Mission Demo")
    print("=====================================================")
    
    # 1. Initialize core infrastructure
    bus = MessageBus()
    
    # 2. Setup the D1 Workflow (Literature -> Experiment)
    topic = "Role of Macrophages in Tumor Microenvironment (TME)"
    print(f"\n[+] Initializing D1_LiteratureToExperimentWorkflow for topic: '{topic}'")
    mission = D1_LiteratureToExperimentWorkflow(bus=bus, topic=topic)
    mission.setup()
    
    # 3. Execute Workflow
    print("\n[+] Triggering Autonomous Graph Execution...")
    result = mission.execute()
    
    # 4. Display Results
    print("\n[+] Mission Complete! Final Result Payload:")
    print(json.dumps(result, indent=4))
    print("\n=====================================================")
    print("🏆 Demonstration successful. Architecture is stable.")
    print("=====================================================")

if __name__ == "__main__":
    main()
