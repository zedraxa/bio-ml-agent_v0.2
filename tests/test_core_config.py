import pytest
import os
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open

from core.config import AgentConfig

def test_default_config():
    """Varsayılan yapılandırma değerlerinin doğru yüklendiğini test et."""
    config = AgentConfig()
    
    assert config.model == "qwen2.5:7b-instruct"
    assert config.max_steps == 9999
    assert config.timeout == 180
    assert config.workspace == Path("workspace")
    assert config.history_dir == Path("conversation_history")

def test_config_from_env_vars(monkeypatch):
    """Ortam değişkenlerinden yapılandırmanın okunmasını test et (Eğer utils.config kullanılıyorsa, burada pydantic davranışı test ediyoruz)."""
    # Pydantic BaseSettings değil BaseModel kullanılıyor, ortam değişkeni parse etmesi için utils.config load_config gerekli
    # Fakat sırf pydantic init'i de test edilebilir.
    config = AgentConfig(model="gpt-4", max_steps=100, timeout=600)
    
    assert config.model == "gpt-4"
    assert config.max_steps == 100
    assert config.timeout == 600

def test_config_from_yaml():
    """Bağımsız pydantic model init (yaml okunmuş varsayımı üzerine)."""
    # config.yaml okuması utils/config.py içinde çalışıyor ama bu core/config testi
    # Biz yaml datası geldikten sonrasını test edeceğiz
    yaml_data = {
        "model": "claude-3",
        "max_steps": 25,
        "timeout": 120,
        "workspace": "test_workspace/my_proj",
        "history_dir": "test_history"
    }
    
    config = AgentConfig(**yaml_data)
    
    assert config.model == "claude-3"
    assert config.max_steps == 25
    assert config.timeout == 120
    assert config.workspace == Path("test_workspace/my_proj")
    assert config.history_dir == Path("test_history")

def test_config_property_updates():
    """Test updating properties."""
    config = AgentConfig()
    config.model = "test-model"
    config.max_steps = 10
    
    assert config.model == "test-model"
    assert config.max_steps == 10
