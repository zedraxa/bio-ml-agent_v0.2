import sys
import pytest
from pathlib import Path

# Proje dizinini path'e ekle
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

@pytest.mark.smoke
def test_import_agent_service():
    """Çekirdek servisin (AgentService) sorunsuz yüklenip başlatılabildiğini test eder."""
    from services.agent_service import AgentService
    service = AgentService(model="qwen2.5:7b-instruct", max_steps=5, timeout=10)
    assert service is not None
    assert service.config.model == "qwen2.5:7b-instruct"

@pytest.mark.smoke
def test_import_api_server():
    """FastAPI/Flask API sunucusunun modüllerinin yüklenebilirliğini test eder."""
    try:
        import api_server
        assert hasattr(api_server, "app")
    except ImportError as e:
        pytest.fail(f"api_server yüklenemedi: {e}")

@pytest.mark.smoke
def test_import_web_ui():
    """Gradio UI'ın syntax/import hatası vermeden fonksiyonunu sunabildiğini test eder."""
    try:
        import web_ui
        assert hasattr(web_ui, "create_ui")
    except ImportError as e:
        pytest.fail(f"web_ui yüklenemedi: {e}")

@pytest.mark.smoke
def test_import_whatsapp_connector():
    """WhatsApp webhook/bot konektörünün import hatalarını kontrol eder."""
    try:
        import whatsapp_connector
        # Whatsapp_connector depends on Flask normally
        assert hasattr(whatsapp_connector, "app")
    except ImportError as e:
        pytest.fail(f"whatsapp_connector yüklenemedi: {e}")

@pytest.mark.smoke
def test_import_cli():
    """CLI scriptinin import ve temel argüman yapısını test eder."""
    try:
        import agent
        assert hasattr(agent, "main")
        assert hasattr(agent, "interactive_shell")
    except ImportError as e:
        pytest.fail(f"agent.py (CLI) yüklenemedi: {e}")

@pytest.mark.smoke
def test_message_normalizer():
    """MessageNormalizer'in import aşamasını ve basit bir metodunu test eder."""
    try:
        from models.messages import MessageNormalizer
        # Basit bir text mesajını OpenAI formatına çevirip test edelim
        test_history = [{"role": "user", "content": "Merhaba"}]
        oai_msg = MessageNormalizer.to_openai(test_history)
        assert len(oai_msg) == 1
        assert oai_msg[0]["role"] == "user"
        assert oai_msg[0]["content"] == "Merhaba"
    except ImportError as e:
        pytest.fail(f"MessageNormalizer yüklenemedi: {e}")
