import os
import base64
import mimetypes
from typing import List, Dict, Union, Any, Tuple

def encode_image_base64(filepath: str) -> str:
    with open(filepath, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class MessageNormalizer:
    """
    Kullanıcının gönderdiği (text + arayüz üzerinden file/image payload'ları barındıran)
    karmaşık mesaj history alanını, farklı modellerin (OpenAI, Anthropic, Gemini, Ollama)
    native API payload türlerine çeviren adaptör katmanıdır.
    """
    
    @staticmethod
    def to_openai(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """OpenAI formatı (gpt-4o / gpt-4o-mini vision desteği ile)."""
        openai_msgs = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            
            if isinstance(content, str):
                openai_msgs.append({"role": role, "content": content})
            elif isinstance(content, list):
                parts = []
                for item in content:
                    if item.get("type") == "text":
                        parts.append({"type": "text", "text": item["text"]})
                    elif item.get("type") == "file":
                        path = item["path"]
                        if not os.path.exists(path):
                            continue
                        mime_type, _ = mimetypes.guess_type(path)
                        
                        # OpenAI Vision desteği doğrudan base64 url istiyor
                        if mime_type and mime_type.startswith("image/"):
                            base64_img = encode_image_base64(path)
                            parts.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_img}"
                                }
                            })
                        else:
                            # OpenAI döküman işlemede (eğer asistan API'si değilsen) dosyayı content'e alamıyor
                            # Basitçe dosya adı metin olarak ekleniyor.
                            parts.append({"type": "text", "text": f"\n[Eklenmiş Dosya: {os.path.basename(path)}]\n"})
                
                if parts:
                    openai_msgs.append({"role": role, "content": parts})
                    
        return openai_msgs

    @staticmethod
    def to_anthropic(messages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """Anthropic formatı (Claude 3 Vision ve PDF analizi)."""
        system_msg = ""
        anthropic_msgs = []
        
        for m in messages:
            role = m["role"]
            content = m["content"]
            
            # Anthropic system mesajlarını ana history'den ayırır
            if role == "system":
                system_msg = content if isinstance(content, str) else "\n".join([i["text"] for i in content if i.get("type") == "text"])
                continue
            
            if isinstance(content, str):
                anthropic_msgs.append({"role": role, "content": content})
            elif isinstance(content, list):
                parts = []
                for item in content:
                    if item.get("type") == "text":
                        parts.append({"type": "text", "text": item["text"]})
                    elif item.get("type") == "file":
                        path = item["path"]
                        if not os.path.exists(path):
                            continue
                        mime_type, _ = mimetypes.guess_type(path)
                        
                        if mime_type and mime_type.startswith("image/"):
                            base64_img = encode_image_base64(path)
                            parts.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime_type,
                                    "data": base64_img
                                }
                            })
                        elif mime_type == "application/pdf":
                            base64_pdf = encode_image_base64(path)
                            parts.append({
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime_type,
                                    "data": base64_pdf
                                }
                            })
                        else:
                            parts.append({"type": "text", "text": f"\n[Eklenmiş Dosya: {os.path.basename(path)}]\n"})
                if parts:
                    anthropic_msgs.append({"role": role, "content": parts})
        
        return system_msg, anthropic_msgs

    @staticmethod
    def to_gemini(messages: List[Dict[str, Any]], client=None) -> Tuple[List[Any], Any, Any]:
        """Gemini formatı (google.genai library türlerine ve Files API yüklemesine dönüşüm)."""
        from google.genai import types
        history = []
        system_instruction = ""
        
        for m in messages:
            role = "user" if m["role"] in ("user", "system") else "model"
            if m["role"] == "system":
                system_instruction = m["content"]
                continue
                
            content = m["content"]
            if isinstance(content, str):
                history.append(types.Content(role=role, parts=[types.Part.from_text(text=content)]))
            elif isinstance(content, list):
                parts = []
                for item in content:
                    if item.get("type") == "text":
                        parts.append(types.Part.from_text(text=item["text"]))
                    elif item.get("type") == "file" and client:
                        path = item["path"]
                        if os.path.exists(path):
                            # Gemini'deki dosyaları harici client üzerinden File API aracılığı ile upload et
                            uploaded = client.files.upload(file=path)
                            parts.append(types.Part.from_uri(file_uri=uploaded.uri, mime_type=uploaded.mime_type))
                if parts:
                    history.append(types.Content(role=role, parts=parts))

        config = types.GenerateContentConfig(
            system_instruction=system_instruction if system_instruction else None,
        )

        if history:
            # Gemini send_message için History payload ve son mesaj payloadı istiyor
            last_msg = history.pop() 
            if last_msg.role == "model":
                history.append(last_msg)
                last_msg_content = ""
            else:
                last_msg_content = last_msg.parts
        else:
            last_msg_content = ""
            
        return history, last_msg_content, config

    @staticmethod
    def to_ollama(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ollama format (Llava vb. vision modelleri image string kabul eder)."""
        ollama_msgs = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            if isinstance(content, str):
                ollama_msgs.append({"role": role, "content": content})
            elif isinstance(content, list):
                text_parts = []
                images = []
                for item in content:
                    if item.get("type") == "text":
                        text_parts.append(item["text"])
                    elif item.get("type") == "file":
                        path = item["path"]
                        if not os.path.exists(path):
                            continue
                        mime_type, _ = mimetypes.guess_type(path)
                        
                        # Ollama, array of base64 images veya image path kabul eder
                        if mime_type and mime_type.startswith("image/"):
                            images.append(path)
                        else:
                            text_parts.append(f"\n[Eklenmiş Dosya: {os.path.basename(path)}]\n")
                
                msg_dict = {
                    "role": role,
                    "content": "\n".join(text_parts)
                }
                if images:
                    msg_dict["images"] = images
                ollama_msgs.append(msg_dict)
                
        return ollama_msgs
