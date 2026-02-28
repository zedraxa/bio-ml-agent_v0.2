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

    def chat_stream(self, messages: List[Dict[str, str]], **kwargs):
        """Streaming modunda yanıt al.
        
        Yield eder text parçalarını (str).
        """
        raise NotImplementedError(f"{self.name} backend henüz streaming desteklemiyor.")

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

    def chat_stream(self, messages: List[Dict[str, str]], **kwargs):
        from exceptions import LLMConnectionError
        try:
            import ollama
            client = ollama.Client(host=self.host)
            response = client.chat(model=self.model, messages=messages, stream=True)
            for chunk in response:
                yield chunk["message"]["content"]
        except ImportError:
            raise LLMConnectionError(
                self.model, details="ollama paketi bulunamadı", suggestion="pip install ollama"
            )
        except Exception as e:
            raise LLMConnectionError(self.model, details=str(e))

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

    def chat_stream(self, messages: List[Dict[str, str]], **kwargs):
        from exceptions import LLMConnectionError
        if not self.api_key:
            raise LLMConnectionError(
                self.model, details="OPENAI_API_KEY ortam değişkeni tanımlı değil"
            )
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                **kwargs,
            )
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except ImportError:
            raise LLMConnectionError(self.model, details="openai paketi bulunamadı")
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

    def chat_stream(self, messages: List[Dict[str, str]], **kwargs):
        from exceptions import LLMConnectionError
        if not self.api_key:
            raise LLMConnectionError(self.model, details="ANTHROPIC_API_KEY ortam değişkeni tanımlı değil")
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)

            system_msg = ""
            chat_messages = []
            for m in messages:
                if m["role"] == "system":
                    system_msg = m["content"]
                else:
                    chat_messages.append(m)

            with client.messages.stream(
                model=self.model,
                max_tokens=4096,
                system=system_msg,
                messages=chat_messages,
                **kwargs,
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except ImportError:
            raise LLMConnectionError(self.model, details="anthropic paketi bulunamadı")
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

    def _prepare_gemini_payload(self, messages, client=None):
        import os
        from google.genai import types
        history = []
        system_instruction = ""
        for m in messages:
            if m["role"] == "system":
                system_instruction = m["content"]
            elif m["role"] == "user" or m["role"] == "assistant":
                role = "user" if m["role"] == "user" else "model"
                if isinstance(m["content"], str):
                    history.append(types.Content(role=role, parts=[types.Part.from_text(text=m["content"])]))
                elif isinstance(m["content"], list):
                    parts = []
                    for item in m["content"]:
                        if item.get("type") == "text":
                            parts.append(types.Part.from_text(text=item["text"]))
                        elif item.get("type") == "file":
                            # Use client to upload the file and get a reference
                            path = item["path"]
                            if os.path.exists(path):
                                uploaded = client.files.upload(file=path)
                                parts.append(types.Part.from_uri(file_uri=uploaded.uri, mime_type=uploaded.mime_type))
                    if parts:
                        history.append(types.Content(role=role, parts=parts))

        config = types.GenerateContentConfig(
            system_instruction=system_instruction if system_instruction else None,
        )

        if history:
            last_msg = history.pop() # Son mesajı alıyoruz
            if last_msg.role == "model":
                history.append(last_msg)
                last_msg_content = ""
            else:
                last_msg_content = last_msg.parts
        else:
            last_msg_content = ""
        
        return history, last_msg_content, config

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
            client = genai.Client(api_key=self.api_key)

            history, last_msg_content, config = self._prepare_gemini_payload(messages, client)
            
            chat = client.chats.create(model=self.model, config=config, history=history)
            response = chat.send_message(last_msg_content)
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

    def chat_stream(self, messages: List[Dict[str, str]], **kwargs):
        from exceptions import LLMConnectionError
        if not self.api_key:
            raise LLMConnectionError(model=self.model, details="GEMINI_API_KEY ortam değişkeni tanımlı değil")
        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)

            history, last_msg_content, config = self._prepare_gemini_payload(messages, client)
            
            chat = client.chats.create(model=self.model, config=config, history=history)
            response = chat.send_message_stream(last_msg_content)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except ImportError:
            raise LLMConnectionError(model=self.model, details="google-genai paketi bulunamadı")
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


def summarize_memory(messages: List[Dict[str, str]], backend: LLMBackend, threshold: int = 15) -> List[Dict[str, str]]:
    """Mesaj geçmişi belirtilen limiti aşarsa LLM'i kullanarak özetler ve bağlam penceresini korur.
    
    Args:
        messages: Mevcut mesaj listesi
        backend: LLM Backend instance
        threshold: Özetlemenin tetikleneceği mesaj sayısı sınırı
        
    Returns:
        Özetlenmiş yeni mesaj listesi
    """
    if len(messages) <= threshold:
        return messages

    # System prompt'unu ayır
    system_msg = None
    chat_msgs = []
    for m in messages:
        if m["role"] == "system":
            system_msg = m
        else:
            chat_msgs.append(m)

    # Son N mesajı koru (bağlamın çok kopmaması için)
    keep_last = 6
    if len(chat_msgs) <= keep_last:
        return messages

    to_summarize = chat_msgs[:-keep_last]
    recent = chat_msgs[-keep_last:]

    summary_prompt = (
        "Lütfen aşağıdaki konuşma geçmişini (yapılan analizleri, kullanılan araçları, "
        "dosya yollarını ve kararları kaybetmeden) çok kısa ve öz bir şekilde özetle.\n\n"
        "GEÇMİŞ:\n"
    )
    for m in to_summarize:
        role = m.get("role", "unknown").upper()
        content = m.get("content", "")
        # Token tasarrufu için çok uzun araç çıktılarını kırpalım
        if len(content) > 1000:
            content = content[:1000] + "... (TRUNCATED)"
        summary_prompt += f"[{role}]: {content}\n\n"

    summary_prompt += "Lütfen sadece özeti Markdown formatında döndür."

    try:
        log.info("Geçmiş %d mesaja ulaştı. Özetleme tetikleniyor...", len(messages))
        summary_text = backend.chat([{"role": "user", "content": summary_prompt}])
        log.info("Özetleme başarıyla tamamlandı.")
    except Exception as e:
        log.warning("Geçmiş özetleme başarısız oldu: %s", e)
        return messages

    new_messages = []
    if system_msg:
        new_messages.append(system_msg)
    
    new_messages.append({
        "role": "assistant", 
        "content": f"**[SİSTEM OTOMATİK ÖZETİ - ÖNCEKİ BAĞLAM]**\n{summary_text}"
    })
    
    new_messages.extend(recent)
    return new_messages


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
