import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Proje kök dizinini ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

from multi_agent import ask_data_engineer, _run_subagent

def test_run_subagent_mocked():
    """_run_subagent fonksiyonunun mock bir LLM ile çalışmasını test et."""
    
    with patch("agent.llm_chat") as mock_chat:
        # LLM her çağrıldığında bu yanıtı dönecek (ilk çağrıda tool kullanımı, ikinci çağrıda normal yanıt simüle edilebilir ama basitlik için direkt dışarıdan yanıt dönüyoruz)
        mock_chat.return_value = "İşlem tamamlandı. Dosya kaydedildi: data/processed/clean.csv"
        
        # Fake bir yapılandırma sağlamak için _cfg() mocklanmalı
        with patch("agent._cfg") as mock_cfg:
            mock_cfg_obj = MagicMock()
            mock_cfg_obj.agent.model = "mock-model"
            mock_cfg_obj.workspace.base_dir = "workspace"
            mock_cfg.return_value = mock_cfg_obj
            
            result = _run_subagent("MOCK_AGENT", "You are mock", "Do something", max_steps=2)
            
            assert "İşlem tamamlandı" in result
            assert mock_chat.called
