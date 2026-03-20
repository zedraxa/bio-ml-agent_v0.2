import sys
import pytest
from pathlib import Path

# Proje dizinini path'e ekle
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

@pytest.mark.smoke
def test_import_agent_service():
    """Çekirdek servisin (AgentService) sorunsuz yüklenip başlatılabildiğini test eder."""
    from bio_ml_agent.services.agent_service import AgentService
    service = AgentService(model="qwen2.5:7b-instruct", max_steps=5, timeout=10)
    assert service is not None
    assert service.config.model == "qwen2.5:7b-instruct"

@pytest.mark.smoke
def test_import_api_server():
    """FastAPI API sunucusunun modüllerinin yüklenebilirliğini test eder."""
    try:
        from bio_ml_agent.api.api_server import app
        assert app is not None
    except ImportError as e:
        pytest.fail(f"api_server yüklenemedi: {e}")

@pytest.mark.smoke
def test_import_web_ui():
    """Gradio UI'ın syntax/import hatası vermeden fonksiyonunu sunabildiğini test eder."""
    try:
        from bio_ml_agent.web_ui import create_ui
        assert create_ui is not None
    except ImportError as e:
        pytest.fail(f"web_ui yüklenemedi: {e}")

@pytest.mark.smoke
def test_import_whatsapp_connector():
    """WhatsApp webhook/bot konektörünün import hatalarını kontrol eder."""
    try:
        from bio_ml_agent.whatsapp_connector import app
        assert app is not None
    except ImportError as e:
        pytest.fail(f"whatsapp_connector yüklenemedi: {e}")

@pytest.mark.smoke
def test_agentservice_modular_components():
    """AgentService'in alt modüllerinin yüklenebilirliğini test eder."""
    try:
        from bio_ml_agent.services.agent.orchestration import get_routing_decision
        from bio_ml_agent.services.agent.memory_context import get_compressed_context
        from bio_ml_agent.services.agent.execution_policy import needs_approval
        from bio_ml_agent.services.agent.project_lifecycle import ensure_project_context
        
        assert all([get_routing_decision, get_compressed_context, needs_approval, ensure_project_context])
    except ImportError as e:
        pytest.fail(f"AgentService alt modülleri yüklenemedi: {e}")

@pytest.mark.smoke
def test_import_cli():
    """CLI scriptinin import ve temel argüman yapısını test eder."""
    try:
        import legacy.agent as agent
        assert hasattr(agent, "main")
    except ImportError as e:
        pytest.fail(f"agent.py (CLI) yüklenemedi: {e}")

@pytest.mark.smoke
def test_message_normalizer():
    """MessageNormalizer'in import aşamasını ve basit bir metodunu test eder."""
    try:
        from bio_ml_agent.models.messages import MessageNormalizer
        # Basit bir text mesajını OpenAI formatına çevirip test edelim
        test_history = [{"role": "user", "content": "Merhaba"}]
        oai_msg = MessageNormalizer.to_openai(test_history)
        assert len(oai_msg) == 1
        assert oai_msg[0]["role"] == "user"
        assert oai_msg[0]["content"] == "Merhaba"
    except ImportError as e:
        pytest.fail(f"MessageNormalizer yüklenemedi: {e}")
