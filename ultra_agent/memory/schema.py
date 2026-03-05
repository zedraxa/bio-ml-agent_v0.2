from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class MemoryEntry(BaseModel):
    """
    S6-1, S6-2: Semantik Hafıza Şeması (Structured Memory Schema).
    Belleği sadece bir metin olmaktan çıkarıp zengin meta-verilerle yapılandırır.
    """
    content: str = Field(..., description="Anının ana içeriği")
    summary: Optional[str] = Field(None, description="Anının kısa özeti")
    memory_type: str = Field("interaction", description="Tip: interaction, decision, artifact, fact, preference")
    source_kind: str = Field("unknown", description="Kaynak türü: chat, terminal, file, thought")
    source_ref: Optional[str] = Field(None, description="Kaynak referansı (örn: dosya yolu, URL)")
    project: str = Field("", description="İlgili proje ID/Workspace")
    session_id: Optional[str] = Field(None, description="Konuşma oturumu ID'si")
    task_id: Optional[str] = Field(None, description="İlgili görev ID'si")
    tags: List[str] = Field(default_factory=list, description="Kategorizasyon etiketleri")
    entities: List[str] = Field(default_factory=list, description="Anıda geçen önemli varlıklar/terimler")
    importance: float = Field(0.5, ge=0.0, le=1.0, description="Anının önemi (0-1)")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Anıya olan güven skoru (0-1)")
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed_at: datetime = Field(default_factory=datetime.now)
    ttl_days: Optional[int] = Field(30, description="Kaç gün saklanacağı")
    provenance: Optional[str] = Field(None, description="Anının neden hatırlandığına dair açıklama")
    version: str = Field("1.0.0", description="Şema versiyonu")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Ekstra serbest alanlar")
