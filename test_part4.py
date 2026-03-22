import os
import sys

# Ensure local imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from bio_ml_agent.swarm.academic_publishing_expert import AcademicPublishingExpertAgent
from bio_ml_agent.swarm.base import SwarmContext

# A mock LLM to prevent actual API calls during logic testing
class MockLLM:
    def chat(self, messages):
        prompt = messages[0]['content']
        if "Analyze the user prompt and decide which academic pipeline to run" in prompt:
            if "deney notları" in prompt:
                return "LAB_REPORT"
            elif "literatür" in prompt:
                return "PAPER_DRAFT"
            elif "hakem" in prompt:
                return "PEER_REVIEW"
            else:
                return "LAB_REPORT"
        
        return '{"section_name": "mock", "content": "mock text", "role_name": "mock_role"}'

import bio_ml_agent.core.agent_base
bio_ml_agent.core.agent_base.auto_create_backend = lambda x: MockLLM()
import bio_ml_agent.swarm.base
bio_ml_agent.swarm.base.auto_create_backend = lambda x: MockLLM()

def test_academic_expert():
    print("Initializing AcademicPublishingExpertAgent...")
    context = SwarmContext(workspace_path="./", model="mock")
    expert = AcademicPublishingExpertAgent(context=context)
    expert.llm = MockLLM() # override with mock
    
    print("\n--- TEST 1: LAB REPORT SENARYOSU ---")
    context.intent = "LAB_REPORT"
    context.shared_memory["last_user_message"] = "Bugün lab dersinde mikroskop ile hücre zarı deneyi yaptık, deney notları ektedir. Bana bir Lab Report yaz."
    
    try:
        result = expert.execute(context)
        print("Success! Result preview:")
        print(result[:150] + "...\n")
    except Exception as e:
        print(f"FAILED Test 1: {e}")

    print("\n--- TEST 2: MAKALE TASLAĞI SENARYOSU ---")
    context.intent = "WRITE_PAPER"
    context.shared_memory["last_user_message"] = "Elimdeki analiz ve literatür verilerini Nature formatında makale draftına çevir."
    
    try:
        result_paper = expert.execute(context)
        print("Success! Result preview:")
        print(result_paper[:150] + "...\n")
    except Exception as e:
        print(f"FAILED Test 2: {e}")

if __name__ == "__main__":
    test_academic_expert()
