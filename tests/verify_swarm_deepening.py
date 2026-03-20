"""Verification script for Multi-agent Swarm Deepening."""
import os
import sys
from unittest.mock import MagicMock, patch

# src dizinini path'e ekle
sys.path.append(os.path.join(os.getcwd(), "src"))

def test_swarm_orchestration():
    print("🚀 Swarm Orchestration Doğrulaması Başlıyor...")
    
    # Mock Config
    mock_cfg = MagicMock()
    mock_cfg.model = "mock-model"
    mock_cfg.workspace = "mock_workspace"
    mock_cfg.swarm = True
    
    # Mock Agents
    from bio_ml_agent.swarm.orchestrator import SwarmOrchestrator
    
    with patch("bio_ml_agent.swarm.researcher.ResearchAgent.execute") as mock_res, \
         patch("bio_ml_agent.swarm.data_engineer.DataEngineerAgent.execute") as mock_de, \
         patch("bio_ml_agent.swarm.ml_expert.MLExpertAgent.execute") as mock_ml, \
         patch("bio_ml_agent.swarm.bioinfo_expert.BioinfoExpertAgent.execute") as mock_bio:
         
        mock_res.return_value = "Literature found high correlation between gene X and cancer Y."
        mock_de.return_value = "cleaned_data.csv"
        mock_ml.return_value = "Model trained. Accuracy 0.92. Feature importance: Gene X (0.8)."
        mock_bio.return_value = "Clinical report: Gene X is a known biomarker for cancer Y..."
        
        orchestrator = SwarmOrchestrator(mock_cfg)
        
        messages = [{"role": "user", "content": "Analyze oncology data for cancer Y and gene X."}]
        
        events = list(orchestrator.process(messages))
        
        # Olayları kontrol et
        found_res = False
        found_de = False
        found_ml = False
        found_bio = False
        found_final = False
        
        for event in events:
            print(f"📡 Event: {event}")
            content = str(event.get("content", ""))
            if "Researcher" in content or "tarıyor" in content: found_res = True
            if "Data Engineer" in content or "temizliyor" in content: found_de = True
            if "ML Expert" in content or "eğitiyor" in content: found_ml = True
            if "Biyoinformatik" in content or "harmanlıyor" in content: found_bio = True
            if event["type"] == "assistant": found_final = True
            
        assert found_res, "Researcher phase missing"
        assert found_de, "Data Engineer phase missing"
        assert found_ml, "ML Expert phase missing"
        assert found_bio, "Bioinfo Expert phase missing"
        assert found_final, "Final assistant response missing"
        
        print("\n✅ Swarm Orchestration başarıyla doğrulandı!")

if __name__ == "__main__":
    try:
        test_swarm_orchestration()
    except Exception as e:
        print(f"❌ Doğrulama hatası: {e}")
        sys.exit(1)
