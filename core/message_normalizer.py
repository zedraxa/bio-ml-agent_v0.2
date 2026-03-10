import base64
from typing import List, Optional, Any, Dict, Union
from pydantic import BaseModel, Field
import json
import logging

logger = logging.getLogger(__name__)

# Standartlaştırılmış Mesaj Formatı
class StandardMessage(BaseModel):
    role: str  # 'system', 'user', 'assistant', 'tool'
    content: Optional[str] = None
    images: Optional[List[str]] = Field(default_factory=list)  # Base64 encoded srtings veya public URL'ler.
    audio: Optional[List[str]] = Field(default_factory=list) # Base64 encoded audio
    files: Optional[List[str]] = Field(default_factory=list) # Dosya yolları
    tool_calls: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    tool_call_id: Optional[str] = None

class MessageNormalizer:
    """
    Tüm giriş (CLI, Web UI, WhatsApp, API) ve çıkış (Gemini, OpenAI, Anthropic) 
    mesajlarını tek bir `StandardMessage` yapısında toplar ve optimize eder.
    """
    
    @staticmethod
    def normalize_input(raw_input: Union[str, Dict[str, Any], List[Dict[str, Any]]], role: str = "user") -> StandardMessage:
        """
        Gelen herhangi bir ham veriyi StandardMessage'a dönüştürür.
        """
        if isinstance(raw_input, str):
            # Düz metin (CLI veya Basit API)
            return StandardMessage(role=role, content=raw_input)
            
        elif isinstance(raw_input, dict):
            # Gradio Multimodal Dict veya OpenAI Chat formatı
            content = raw_input.get("text") or raw_input.get("content")
            
            # Gradio files yapısı (Eğer varsa)
            files = []
            if "files" in raw_input and isinstance(raw_input["files"], list):
                for f in raw_input["files"]:
                    if isinstance(f, dict) and "path" in f:
                         files.append(f["path"])
                    elif isinstance(f, str):
                         files.append(f)
                         
            return StandardMessage(
                role=role,
                content=str(content) if content else None,
                files=files
            )
            
        elif isinstance(raw_input, list):
            # Langchain veya Anthropic complex blok yapısı
            content_parts = []
            images = []
            files = []
            
            for part in raw_input:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        content_parts.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        if url: images.append(url)
                    elif part.get("type") == "file":
                         url = part.get("file_url", {}).get("url", "")
                         if url: files.append(url)
                elif isinstance(part, str):
                     content_parts.append(part)
                     
            return StandardMessage(
                role=role,
                content="\n".join(content_parts) if content_parts else None,
                images=images,
                files=files
            )
            
        else:
            logger.warning(f"Unrecognized input format: {type(raw_input)}")
            return StandardMessage(role=role, content=str(raw_input))
            
    @staticmethod
    def to_provider_format(message: StandardMessage, provider: str = "openai") -> Dict[str, Any]:
         """
         StandardMessage'ı hedef LLM sağlayıcısının beklediği özel formata (Dict) dönüştürür.
         """
         provider = provider.lower()
         
         if provider in ["openai", "ollama", "groq"]:
              # Standart OpenAI formatı
              if not message.images and not message.files:
                   return {"role": message.role, "content": message.content or ""}
              
              # Gelişmiş (Multimodal) format
              content_array = []
              if message.content:
                   content_array.append({"type": "text", "text": message.content})
              for img in message.images:
                   # Varsayım: Base64 data URI veya direkt URL geliyor.
                   url = img if img.startswith(("http", "data:")) else f"data:image/jpeg;base64,{img}"
                   content_array.append({"type": "image_url", "image_url": {"url": url}})
                   
              return {"role": message.role, "content": content_array}
              
         elif provider == "anthropic":
             # Anthropic messages API formatı
             res = {"role": message.role}
             if message.images:
                  content_array = []
                  if message.content:
                       content_array.append({"type": "text", "text": message.content})
                  for img in message.images:
                       # Anthropic base64 format beklentisi:
                       clean_b64 = img.split(",")[-1] if "base64," in img else img
                       content_array.append({
                           "type": "image",
                           "source": { "type": "base64", "media_type": "image/jpeg", "data": clean_b64 }
                       })
                  res["content"] = content_array
             else:
                  res["content"] = message.content or ""
             return res
             
         elif provider == "gemini":
              # Google Generative AI formatı
              role_map = {"system": "user", "user": "user", "assistant": "model", "tool": "function"}
              g_role = role_map.get(message.role, "user")
              
              parts = []
              if message.content:
                  parts.append(message.content)
                  
              # Not: Gemini File API yerel dosya yüklemesi gerektirir. 
              # Burada sadece text varsayımını koruyoruz, ileriki sprintlerde genişletilecek.
              return {"role": g_role, "parts": parts}
              
         else:
             # Default Fallback (Langchain stili sade)
             return {"role": message.role, "content": message.content or ""}
