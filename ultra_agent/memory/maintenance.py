import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

log = logging.getLogger("bio_ml_agent")

def synthesize_memories_llm(memories: List[Dict[str, Any]], project: str, model_name: str = "gemini-2.0-flash") -> Optional[Dict[str, Any]]:
    """
    Verilen benzer anı listesini LLM kullanarak tek bir kapsayıcı anı haline getirir.
    Faz 9 - Memory Maintenance: Merge
    """
    if len(memories) < 2:
        return None
        
    try:
        from llm_backend import auto_create_backend
        llm = auto_create_backend(model_name=model_name)
    except Exception as e:
        log.error(f"LLM Backend yüklenemedi: {e}")
        return None
        
    prompt = f"""
Sen bir 'Hafıza Birleştirici' (Memory Synthesizer) ajansın.
Aşağıda aynı proje ({project}) içindeki, birbirine anlamsal olarak çok benzer olan veya birbirini tamamlayan eski parçalı anıların listesi verilmiştir.
Görevin, bu anıları analiz edip bilgiyi kaybetmeden Hepsini TEK bir özet anı altında birleştirmektir.

## Kurallar:
1. Birleştirilmiş yeni anı (Merged Memory), tüm parçalardaki önemli teknik detayları, kararları ve gerçekleri (fact) barındırmalıdır.
2. Çelişen bilgiler varsa en güncel olanına (Recency) güvenilmelidir.
3. Sonucu SADECE GEÇERLİ BİR JSON formatında dönmelisin. Ekstra raw metin içermemelidir.

## Parçalı Anılar:
"""
    for i, mem in enumerate(memories):
        date_str = mem.get("created_at", "Bilinmiyor")
        content = mem.get("content", "")
        summary = mem.get("summary", "")
        prompt += f"\n--- Anı {i+1} (Tarih: {date_str}) ---\nİçerik: {content}\nÖzet: {summary}\n"
        
    prompt += """

## JSON Çıktı Formatı:
{
    "content": "Birleştirilmiş, detaylı ve tutarlı ana içerik.",
    "summary": "Kısa ve net bir özet.",
    "memory_type": "interaction",  // artifact, decision, fact, interaction'dan biri
    "tags": ["etiket1", "etiket2", "merged"],
    "entities": ["Varlık1", "Varlık2"],
    "importance": 0.8 // 0.0 - 1.0 arası genel önem skoru
}
"""

    try:
        response_text = llm.chat([
            {"role": "system", "content": "You are a specialized JSON-outputting memory synthesizer agent."},
            {"role": "user", "content": prompt}
        ])
        
        # Olası Markdown kod bloklarını temizle
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
            
        merged_data = json.loads(response_text.strip())
        
        # Temel validasyon
        if "content" not in merged_data:
            raise ValueError("LLM yanıtında 'content' alanı bulunamadı.")
            
        merged_data["created_at"] = datetime.now().isoformat()
        merged_data["last_accessed_at"] = datetime.now().isoformat()
        merged_data["project"] = project
        merged_data["source_kind"] = "merged"
        merged_data["provenance"] = f"{len(memories)} adet anının sentezlenmesi ile oluşturuldu."
        
        # Eski kaynak referanslarını birleştir
        refs = [m.get("source_ref") for m in memories if m.get("source_ref")]
        if refs:
            merged_data["source_ref"] = ", ".join(set(refs))
            
        # Eski ID leri kaydet (silinmeleri gerekecek)
        merged_data["metadata"] = {
            "merged_from_ids": [str(m.get("id")) for m in memories]
        }
        
        return merged_data
        
    except Exception as e:
        log.error(f"Memories merge işlemi başarsız: {e}")
        return None

class MemoryMerger:
    """Qdrant Store içinde birbirine benzeyen hafıza kayıtlarını bulan ve sentezleyen sınıf."""
    def __init__(self, store: Any, model_name: str = "gemini-2.0-flash"):
        self.store = store
        self.model_name = model_name
        
    def merge_project_memories(self, project: str, similarity_threshold: float = 0.90) -> int:
        """
        Belirtilen proje içindeki yüksek benzerlikli anı kümelerini bulur,
        Birleştirir ve eski (parçalı) anıları siler.
        Kaç adet yeni/sentez anı yaratıldığını döndürür.
        """
        if not self.store.enabled:
            return 0
            
        memories = self.store.list_memories_for_project(project, limit=500)
        if len(memories) < 2:
            return 0
            
        merged_count = 0
        processed_ids = set()
        
        # TODO: Optimal bir kümeleme (Clustering) algoritması kullanılabilirdi, 
        # Şu an basit bir dolaşma yapıp birbirine çok benzer olanları grup olarak ele alacağız
        
        from ultra_agent.memory.qdrant_store import _encode_text
        
        clusters = []
        for mem in memories:
            mem_id = str(mem.get("id"))
            if mem_id in processed_ids:
                continue
                
            mem_content = mem.get("content", "")
            if not mem_content or len(mem_content) < 20:
                continue
                
            # Benzerleri bul: Kendi yazdığımız arama vektörüyle
            similar_results = self.store.search_memory(
                query=mem_content, 
                limit=10, 
                min_score=similarity_threshold,
                project_filter=project
            )
            
            # Sadece cluster kurmaya değerse (kendisinden başka bir şey bulduysa)
            cluster_members = []
            for res in similar_results:
                res_id = str(res.get("id"))
                if res_id not in processed_ids:
                    cluster_members.append(res)
                    
            if len(cluster_members) >= 2: # Kendisi + en az 1 başka benzeri
                clusters.append(cluster_members)
                for member in cluster_members:
                    processed_ids.add(str(member.get("id")))
            else:
                processed_ids.add(mem_id)
                
        # Bulunan kümeler LLM ile sentezle
        for cluster in clusters:
            log.info(f"Yüksek benzerlikli {len(cluster)} anı birleştirilecek.")
            merged_payload = synthesize_memories_llm(cluster, project, self.model_name)
            
            if merged_payload:
                # 1. Yeni (Sentez) anıyı kaydet
                try:
                    import uuid
                    new_id = str(uuid.uuid4())
                    
                    from ultra_agent.memory.schema import MemoryEntry
                    entry = MemoryEntry(**merged_payload)
                    
                    vector = _encode_text(entry.content)
                    
                    from qdrant_client.models import PointStruct
                    point = PointStruct(
                        id=new_id,
                        vector=vector,
                        payload=entry.dict()
                    )
                    
                    self.store.client.upsert(
                        collection_name=self.store.collection_name,
                        points=[point]
                    )
                    log.info(f"Sentezlenmiş hafıza eklendi -> ID: {new_id}")
                    
                    # 2. Eskileri Sil
                    old_ids = merged_payload["metadata"]["merged_from_ids"]
                    for old_id in old_ids:
                        self.store.delete_memory(old_id)
                        log.debug(f"Eski anı silindi -> ID: {old_id}")
                        
                    merged_count += 1
                except Exception as e:
                    log.error(f"Sentezlenmiş hafızanın Qdrant'a kaydı sırasında hata: {e}")
                    
        return merged_count
