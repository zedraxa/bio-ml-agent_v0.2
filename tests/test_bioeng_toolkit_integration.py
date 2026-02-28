import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import llm_chat
from bioeng_toolkit import ProteinAnalyzer

def test_bioeng_integration():
    """
    Bio-ML Ajanının biyo-mühendislik araç setini kullanabilecek şekilde tasarlandığını doğrular.
    Gerçekten LLM çağırmayacağız ancak orkestra yönergelerine eklediğimiz `bioeng_toolkit`'in 
    çağrılabildiğini basite indirgenmiş mock olarak doğruluyoruz.
    """
    
    # 1. Gerçek tool'un kullanımını test ediyoruz (Agent in python kodunu simüle edelim)
    p = ProteinAnalyzer("MKWVTFISLL")
    assert p.length == 10
    assert "M" in getattr(p, "amino_acid_composition", lambda: {})() or p.molecular_weight() > 0
    
    # 2. Agent'ın prompt içerisinde bioeng_toolkit talimatına sahip olup olmadığını test etsek de olur 
    # ama statik test yeterli. Sadece doğru yerleştirildiğini onaylamak için promptu basitçe parselliyoruz:
    from agent import SYSTEM_PROMPT
    assert "BIOENGINEERING TOOLKIT" in SYSTEM_PROMPT
    assert "from bioeng_toolkit import" in SYSTEM_PROMPT
    assert "ProteinAnalyzer" in SYSTEM_PROMPT
    assert "MedicalImageHelper" in SYSTEM_PROMPT
