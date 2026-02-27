# llm_backend.py
# ═══════════════════════════════════════════════════════════
#  Bio-ML Agent — Çoklu LLM Backend Desteği
#  Ollama, OpenAI, Anthropic ve Gemini backend'lerini destekler.
# ═══════════════════════════════════════════════════════════

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

log = logging.getLogger("bio_ml_agent")


# ─────────────────────────────────────────────
#  Abstract Base Class
# ─────────────────────────────────────────────

class LLMBackend(ABC):
    """Tüm LLM backend'lerin temel sınıfı.

    Her backend `chat()` metodunu uygulamalıdır.
    """

    name: str = "base"

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Mesaj listesi gönderip yanıt al.

        Args:
            messages: OpenAI formatında mesaj listesi
                      [{"role": "system"|"user"|"assistant", "content": "..."}]
        Returns:
            Asistan yanıtı (str).
        Raises:
            LLMConnectionError: Bağlantı veya API hatası.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Backend'in kullanılabilir olup olmadığını kontrol et."""
        ...

    def list_models(self) -> List[str]:
        """Mevcut modellerin listesini döndür (destekleniyorsa)."""
        return []

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"


# ─────────────────────────────────────────────
#  Ollama Backend (Yerel)
# ─────────────────────────────────────────────

class OllamaBackend(LLMBackend):
    """Yerel Ollama sunucusu üzerinden LLM çağrısı.

    Varsayılan olarak http://localhost:11434 adresini kullanır.
    """

    name = "ollama"

    def __init__(self, model: str = "qwen2.5:latest", host: Optional[str] = None):
        self.model = model
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        from exceptions import LLMConnectionError
        try:
            import ollama
            client = ollama.Client(host=self.host)
            response = client.chat(model=self.model, messages=messages)
            return response["message"]["content"]
        except ImportError:
            raise LLMConnectionError(
                self.model,
                details="ollama paketi bulunamadı",
                suggestion="pip install ollama",
            )
        except Exception as e:
            raise LLMConnectionError(
                self.model, details=str(e),
                suggestion="Ollama servisinin çalıştığından emin olun: ollama serve",
            )

    def is_available(self) -> bool:
        try:
            import ollama
            client = ollama.Client(host=self.host)
            client.list()
            return True
        except Exception:
            return False

    def list_models(self) -> List[str]:
        try:
            import ollama
            client = ollama.Client(host=self.host)
            models = client.list()
            return [m["name"] for m in models.get("models", [])]
        except Exception:
            return []


# ─────────────────────────────────────────────
#  OpenAI Backend
# ─────────────────────────────────────────────

class OpenAIBackend(LLMBackend):
    """OpenAI API (GPT-4, GPT-3.5-turbo, vb.).

    API key: OPENAI_API_KEY ortam değişkeninden okunur.
    """

    name = "openai"

    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        from exceptions import LLMConnectionError
        if not self.api_key:
            raise LLMConnectionError(
                self.model,
                details="OPENAI_API_KEY ortam değişkeni tanımlı değil",
                suggestion="export OPENAI_API_KEY='sk-...' komutunu çalıştırın.",
            )
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs,
            )
            return response.choices[0].message.content
        except ImportError:
            raise LLMConnectionError(
                self.model,
                details="openai paketi bulunamadı",
                suggestion="pip install openai",
            )
        except Exception as e:
            raise LLMConnectionError(self.model, str(e))

    def is_available(self) -> bool:
        return bool(self.api_key)

    def list_models(self) -> List[str]:
        return ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"]


# ─────────────────────────────────────────────
#  Anthropic Backend
# ─────────────────────────────────────────────

class AnthropicBackend(LLMBackend):
    """Anthropic API (Claude 3, Claude 3.5, vb.).

    API key: ANTHROPIC_API_KEY ortam değişkeninden okunur.
    """

    name = "anthropic"

    def __init__(self, model: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        from exceptions import LLMConnectionError
        if not self.api_key:
            raise LLMConnectionError(
                self.model,
                details="ANTHROPIC_API_KEY ortam değişkeni tanımlı değil",
                suggestion="export ANTHROPIC_API_KEY='sk-ant-...' komutunu çalıştırın.",
            )
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)

            # Anthropic API system mesajı ayrı parametre olarak alır
            system_msg = ""
            chat_messages = []
            for m in messages:
                if m["role"] == "system":
                    system_msg = m["content"]
                else:
                    chat_messages.append(m)

            response = client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_msg,
                messages=chat_messages,
                **kwargs,
            )
            return response.content[0].text
        except ImportError:
            raise LLMConnectionError(
                self.model,
                details="anthropic paketi bulunamadı",
                suggestion="pip install anthropic",
            )
        except Exception as e:
            raise LLMConnectionError(self.model, str(e))

    def is_available(self) -> bool:
        return bool(self.api_key)

    def list_models(self) -> List[str]:
        return [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
        ]


# ─────────────────────────────────────────────
#  Google Gemini Backend
# ─────────────────────────────────────────────

class GeminiBackend(LLMBackend):
    """Google Gemini API.

    API key: GEMINI_API_KEY ortam değişkeninden okunur.
    """

    name = "gemini"

    def __init__(self, model: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        from exceptions import LLMConnectionError
        if not self.api_key:
            raise LLMConnectionError(
                model=self.model,
                message="GEMINI_API_KEY ortam değişkeni tanımlı değil",
                details="GEMINI_API_KEY ortam değişkeni tanımlı değil",
                suggestion="export GEMINI_API_KEY='...' komutunu çalıştırın.",
            )
        try:
            from google import genai
            from google.genai import types
            
            client = genai.Client(api_key=self.api_key)

            # OpenAI formatını Gemini formatına çevir
            history = []
            system_instruction = ""
            for m in messages:
                if m["role"] == "system":
                    system_instruction = m["content"]
                elif m["role"] == "user":
                    history.append(types.Content(role="user", parts=[types.Part.from_text(text=m["content"])]))
                elif m["role"] == "assistant":
                    history.append(types.Content(role="model", parts=[types.Part.from_text(text=m["content"])]))

            config = types.GenerateContentConfig(
                system_instruction=system_instruction if system_instruction else None,
            )

            if history:
                 last_msg = history.pop() # Son mesajı alıyoruz
                 if last_msg.role == "model":
                     # Eğer son mesaj asistan mesajıysa, history'ye geri ekleyip boş mesaj yollayamayız.
                     # Kullanıcıdan gelen son mesaj olması beklenir. Eğer değilse boş yollanır. (gemini buna izin vermeyebilir)
                     # Basitçe tüm history'i verip 'Continue' diyebiliriz ama normal akışta son mesaj user olur.
                     history.append(last_msg)
                     last_msg_text = ""
                 else:
                     last_msg_text = last_msg.parts[0].text
            else:
                 last_msg_text = ""
                 
            chat = client.chats.create(model=self.model, config=config, history=history)
            response = chat.send_message(last_msg_text)
            return response.text
        except ImportError:
            raise LLMConnectionError(
                model=self.model,
                message="google-genai paketi bulunamadı",
                details="google-genai paketi bulunamadı",
                suggestion="pip install google-genai",
            )
        except Exception as e:
            raise LLMConnectionError(self.model, str(e))

    def is_available(self) -> bool:
        return bool(self.api_key)

    def list_models(self) -> List[str]:
        return ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash-lite"]


# ─────────────────────────────────────────────
#  Backend Registry & Factory
# ─────────────────────────────────────────────

_BACKENDS: Dict[str, type] = {
    "ollama": OllamaBackend,
    "openai": OpenAIBackend,
    "anthropic": AnthropicBackend,
    "gemini": GeminiBackend,
}


def create_backend(name: str, **kwargs) -> LLMBackend:
    """İsme göre backend oluştur.

    Args:
        name: Backend adı ("ollama", "openai", "anthropic", "gemini")
        **kwargs: Backend'e özel parametreler (model, api_key, vb.)

    Returns:
        LLMBackend instance.

    Raises:
        ValueError: Bilinmeyen backend adı.
    """
    cls = _BACKENDS.get(name.lower())
    if cls is None:
        available = ", ".join(sorted(_BACKENDS.keys()))
        raise ValueError(
            f"Bilinmeyen LLM backend: {name!r}. "
            f"Desteklenen backend'ler: {available}"
        )
    return cls(**kwargs)


def list_backends() -> List[str]:
    """Desteklenen backend isimlerini döndür."""
    return sorted(_BACKENDS.keys())


def register_backend(name: str, cls: type) -> None:
    """Yeni bir backend kaydet.

    Args:
        name: Backend adı.
        cls: LLMBackend alt sınıfı.
    """
    if not issubclass(cls, LLMBackend):
        raise TypeError(f"{cls!r} LLMBackend alt sınıfı olmalıdır.")
    _BACKENDS[name.lower()] = cls


# ─────────────────────────────────────────────
#  Otomatik Backend Seçimi (local / remote)
# ─────────────────────────────────────────────

# Model adı desenleri → backend eşleştirmesi
_MODEL_PATTERNS: Dict[str, str] = {
    "gpt-": "openai",
    "o1-": "openai",
    "o3-": "openai",
    "o4-": "openai",
    "chatgpt-": "openai",
    "claude-": "anthropic",
    "gemini-": "gemini",
}


def detect_backend_name(model: str) -> str:
    """Model adından backend ismini tahmin et.

    Args:
        model: Model adı (ör: 'gpt-4o-mini', 'claude-3-5-sonnet-20241022', 'gemini-2.0-flash')

    Returns:
        Backend adı ('openai', 'anthropic', 'gemini', 'ollama')
    """
    model_lower = model.lower().strip()
    for prefix, backend_name in _MODEL_PATTERNS.items():
        if model_lower.startswith(prefix):
            return backend_name
    return "ollama"


def auto_create_backend(model: str, mode: str = "auto") -> LLMBackend:
    """Model adı ve moda göre otomatik backend oluştur.

    Args:
        model: Model adı (ör: 'gpt-4o-mini', 'qwen2.5:7b-instruct')
        mode:
            'local'  → Her zaman Ollama kullan
            'remote' → Model adından backend algıla (gpt→OpenAI, claude→Anthropic, gemini→Gemini)
            'auto'   → Model adı bulut sağlayıcısına benziyorsa remote, değilse local

    Returns:
        LLMBackend instance.
    """
    mode = mode.lower().strip()

    if mode == "local":
        log.info("🏠 Backend modu: LOCAL → Ollama | model=%s", model)
        return OllamaBackend(model=model)

    # remote veya auto → model adından backend belirle
    backend_name = detect_backend_name(model)

    if mode == "auto" and backend_name == "ollama":
        log.info("🏠 Backend modu: AUTO → Ollama (yerel model) | model=%s", model)
        return OllamaBackend(model=model)

    if backend_name == "ollama" and mode == "remote":
        # Kullanıcı remote dedi ama model adı yerel gibi görünüyor
        log.warning(
            "⚠️ Backend modu REMOTE ama model '%s' bir bulut modeline benzemiyor. "
            "Yine de Ollama ile denenecek. Bulut API kullanmak için "
            "gpt-4o-mini / claude-3-5-sonnet-20241022 / gemini-2.0-flash gibi model adları kullanın.",
            model,
        )
        return OllamaBackend(model=model)

    log.info("☁️  Backend modu: %s → %s | model=%s", mode.upper(), backend_name.upper(), model)
    return create_backend(backend_name, model=model)
