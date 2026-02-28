import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "workspace" / ".rag_db"

class MemoryManager:
    """
    Kalıcı Uzun Dönem Hafıza (Vector DB RAG)
    AJAN'ın önceki konuşmalarını ve projelerini aylar sonra bile hatırlamasını sağlar.
    """
    
    def __init__(self):
        self.enabled = CHROMA_AVAILABLE
        self.collection = None
        
        if self.enabled:
            import logging
            # Suppress chromadb spam
            logging.getLogger("chromadb").setLevel(logging.ERROR)
            
            DB_DIR.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=str(DB_DIR),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Use a lightweight multi-lingual sentence transformer
            self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="paraphrase-multilingual-MiniLM-L12-v2"
            )
            
            self.collection = self.client.get_or_create_collection(
                name="agent_memory",
                embedding_function=self.emb_fn,
                metadata={"hnsw:space": "cosine"}
            )
            
    def store_interaction(self, session_id: str, user_query: str, agent_response: str, metadata: Optional[Dict] = None):
        """Kullanıcının sorusunu ve ajanın yanıtını vektör veritabanına kaydeder."""
        if not self.enabled or not self.collection:
            return
            
        doc_id = f"{session_id}_{os.urandom(4).hex()}"
        
        # Combine query and response to establish full context embedding
        combined_text = f"USER: {user_query}\nAGENT: {agent_response}"
        
        meta = metadata or {}
        meta["session_id"] = session_id
        meta["type"] = "interaction"
        
        try:
            self.collection.add(
                documents=[combined_text],
                metadatas=[meta],
                ids=[doc_id]
            )
        except Exception as e:
            print(f"[MemoryManager] Hata: Etkileşim kaydedilemedi - {e}")

    def search_memory(self, query: str, n_results: int = 3, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Verilen sorguya anlamsal olarak en çok benzeyen eski konuşmaları getirir."""
        if not self.enabled or not self.collection:
            return []
            
        # Eğer collection boşsa query hata verebilir
        try:
            if self.collection.count() == 0:
                return []
        except Exception:
            return []
            
        where_filter = {"session_id": session_id} if session_id else None
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, self.collection.count()),
                where=where_filter
            )
            
            memories = []
            if results and results.get("documents") and len(results["documents"]) > 0:
                docs = results["documents"][0]
                metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
                dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
                
                for doc, meta, dist in zip(docs, metas, dists):
                    memories.append({
                        "text": doc,
                        "metadata": meta,
                        "distance": dist  # Lower is better (cosine distance)
                    })
                    
            return memories
        except Exception as e:
            print(f"[MemoryManager] Hata: Bellek araması başarısız - {e}")
            return []
            
    def get_context_string(self, query: str, n_results: int = 3) -> str:
        """LLM promptuna enjekte edilmek üzere biçimlendirilmiş hafıza stringi döndürür."""
        memories = self.search_memory(query, n_results=n_results)
        if not memories:
            return ""
            
        context = "Geçmiş konuşmalardan hatırladıkların (Kalıcı Bellek - RAG):\n"
        context += "-" * 50 + "\n"
        for i, mem in enumerate(memories):
            context += f"Anı {i+1} (Alaka Skoru: {1.0 - mem['distance']:.2f}):\n{mem['text']}\n"
            context += "-" * 50 + "\n"
            
        return context

# Singleton instance
memory = MemoryManager()
