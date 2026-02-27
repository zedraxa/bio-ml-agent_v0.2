# tests/conftest.py
# ═══════════════════════════════════════════════════════════
#  Bio-ML Agent — Test Fixture'ları (Paylaşılan)
# ═══════════════════════════════════════════════════════════

import os
import sys
from pathlib import Path

import pytest

# agent.py'yi import edebilmek için proje kökünü sys.path'e ekle
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def tmp_workspace(tmp_path):
    """Testler için geçici bir workspace klasörü oluşturur."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def tmp_history_dir(tmp_path):
    """Testler için geçici bir conversation_history klasörü oluşturur."""
    hd = tmp_path / "conversation_history"
    hd.mkdir()
    return hd


@pytest.fixture
def sample_messages():
    """Örnek konuşma mesajları döndürür."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Merhaba, bana yardım et"},
        {"role": "assistant", "content": "Tabii, nasıl yardımcı olabilirim?"},
        {"role": "user", "content": "Bir ML projesi oluştur"},
        {"role": "assistant", "content": "Hemen başlıyorum!"},
    ]


@pytest.fixture(autouse=True)
def clean_env():
    """Her testten önce/sonra AGENT_PROJECT env değişkenini temizle."""
    old = os.environ.pop("AGENT_PROJECT", None)
    yield
    if old is not None:
        os.environ["AGENT_PROJECT"] = old
    else:
        os.environ.pop("AGENT_PROJECT", None)
