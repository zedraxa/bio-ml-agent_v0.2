import os
from pathlib import Path
from typing import List, Dict, Tuple
import chromadb
from chromadb.config import Settings
import logging

log = logging.getLogger(__name__)

class RAGEngine:
    """Retrieval-Augmented Generation (RAG) motoru.
    Geliştirilmiş projelerin içeriğini (md, txt, py) indeksler ve benzerlik araması sunar.
    """
    def __init__(self, workspace_dir: Path, db_dir_name: str = ".rag_db"):
        self.workspace_dir = Path(workspace_dir)
        self.db_dir = self.workspace_dir / db_dir_name
        self.db_dir.mkdir(parents=True, exist_ok=True)
        
        # ChromaDB Persistent Client
        self.client = chromadb.PersistentClient(path=str(self.db_dir))
        
        # Collection for documents
        self.collection = self.client.get_or_create_collection(
            name="bio_ml_agent_docs",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.supported_extensions = {".md", ".txt", ".py", ".csv", ".json"}
        
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Metni parçalara (chunk) ayırır."""
        chunks = []
        if not text:
            return chunks
            
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            
            # Kelime bölünmesini önlemek için boşluğa kadar git
            if end < text_len and text[end] not in (' ', '\n', '\t'):
                # Geriye dönük son boşluğu bul
                last_space = text.rfind(' ', start, end)
                if last_space != -1 and last_space > start:
                    end = last_space
                    
            chunks.append(text[start:end])
            start = end - overlap
            if start < 0:
                break
                
        return chunks

    def index_workspace(self) -> int:
        """Workspace altındaki tüm desteklenen dosyaları bulur ve indeksler.
        Eskileri temizler ve baştan indeksler.
        
        Döndürdüğü değer işlenen doküman sayısıdır."""
        
        log.info("RAG indekslemesi başlatılıyor...")
        
        # Mevcut collection'ı sil ve yeniden oluştur (tam index yenileme)
        try:
            self.client.delete_collection("bio_ml_agent_docs")
        except Exception:
            pass
            
        self.collection = self.client.get_or_create_collection(
            name="bio_ml_agent_docs",
            metadata={"hnsw:space": "cosine"}
        )
        
        docs = []
        metadatas = []
        ids = []
        
        doc_count = 0
        
        # Tüm dosyaları dolaş
        for root_str, dirs, files in os.walk(self.workspace_dir):
            root = Path(root_str)
            
            # RAG DB dizinini atla
            if self.db_dir.name in root.parts or ".git" in root.parts or "venv" in root.parts:
                continue
                
            for file in files:
                file_path = root / file
                if file_path.suffix.lower() not in self.supported_extensions:
                    continue
                    
                # Çok büyük dosyaları dışla (Örn: >= 1MB csv)
                if file_path.stat().st_size > 1024 * 1024:
                    continue
                    
                try:
                    content = file_path.read_text(encoding="utf-8")
                    chunks = self._chunk_text(content)
                    
                    rel_path = str(file_path.relative_to(self.workspace_dir))
                    
                    for i, chunk in enumerate(chunks):
                        if not chunk.strip():
                            continue
                        
                        docs.append(chunk)
                        metadatas.append({
                            "source": rel_path,
                            "chunk_index": i
                        })
                        ids.append(f"{rel_path}_{i}")
                        
                    doc_count += 1
                except Exception as e:
                    log.warning(f"RAG indeksleme hatası ({file_path}): {e}")
                    
        # ChromaDB'ye ekle (batch halinde)
        if docs:
            batch_size = 100
            for i in range(0, len(docs), batch_size):
                self.collection.add(
                    documents=docs[i:i+batch_size],
                    metadatas=metadatas[i:i+batch_size],
                    ids=ids[i:i+batch_size]
                )
                
        log.info(f"RAG indeksleme tamamlandı. {doc_count} dosya işlendi, {len(docs)} parça eklendi.")
        return doc_count

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Verilen sorguya en benzer metin parçalarını döndürür.
        
        Returns:
            List of dicts containing 'document', 'source', 'distance'
        """
        if not query.strip():
            return []
            
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            if not results['documents'] or not results['documents'][0]:
                return []
                
            formatted_results = []
            for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
                formatted_results.append({
                    "document": doc,
                    "source": meta["source"] if meta else "unknown",
                    "distance": dist
                })
                
            return formatted_results
        except Exception as e:
            log.error(f"RAG arama hatası: {e}")
            return []
