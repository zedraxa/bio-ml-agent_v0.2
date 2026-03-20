import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from bio_ml_agent.ml.bioeng_toolkit import ProteinAnalyzer

def test_bioeng_integration():
    """
    Bio-ML Ajanının biyo-mühendislik araç setini kullanabilecek şekilde tasarlandığını doğrular.
    Gerçekten LLM çağırmayacağız ancak bioeng_toolkit'in çağrılabildiğini doğruluyoruz.
    """
    
    # 1. Gerçek tool'un kullanımını test ediyoruz
    p = ProteinAnalyzer("MKWVTFISLL")
    assert p.length == 10
    assert p.molecular_weight() > 0
    
    # 2. Agent'ın system prompt'unda bioeng_toolkit talimatına sahip olduğunu kontrol et
    from bio_ml_agent.core.config import SYSTEM_PROMPT
    assert "BIOENGINEERING TOOLKIT" in SYSTEM_PROMPT
    assert "ProteinAnalyzer" in SYSTEM_PROMPT
    assert "MedicalImageHelper" in SYSTEM_PROMPT
