import sys
from pathlib import Path

# Proje kökünü ekle
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from bio_ml_agent.core.deep_research import DeepResearchAgent

def test_research():
    agent = DeepResearchAgent(
        model_name="gemini-2.5-flash",
        workspace=Path("workspace"),
        project_name="test_research_project"
    )
    
    query = "Latest advancements in CRISPR-based gene therapies for beta-thalassemia 2024-2025"
    print(f"Starting research for: {query}")
    
    try:
        # Depth=1 for quick test
        # We need to mock tools if we don't want real web access, 
        # but here we want to see if it works end-to-end.
        result = agent.research(query)
        print("\nResearch Result:\n")
        print(result)
    except Exception as e:
        print(f"Error during research: {e}")

if __name__ == "__main__":
    test_research()
