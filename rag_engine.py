import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from bs4 import BeautifulSoup

try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import docx
except ImportError:
    docx = None

try:
    import pptx
except ImportError:
    pptx = None

try:
    import openpyxl
except ImportError:
    openpyxl = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

import json

log = logging.getLogger(__name__)

class RAGEngine:
    """Retrieval-Augmented Generation (RAG) motoru.
    Geliştirilmiş projelerin içeriğini (md, txt, py) indeksler ve benzerlik araması sunar.
    """
    def __init__(self, workspace_dir: Path, db_dir_name: str = ".rag_db"):
        self.workspace_dir = Path(workspace_dir)
        self.db_dir = self.workspace_dir / db_dir_name
        self.db_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            import chromadb
            from chromadb.config import Settings
            # ChromaDB Persistent Client
            self.client = chromadb.PersistentClient(
                path=str(self.db_dir),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Collection for documents
            self.collection = self.client.get_or_create_collection(
                name="bio_ml_agent_docs",
                metadata={"hnsw:space": "cosine"}
            )
        except ImportError:
            log.warning("chromadb yüklü değil, RAGEngine vektör arama yapamayacaktır.")
            self.client = None
            self.collection = None
        
        self.supported_extensions = {
            ".md", ".txt", ".py", ".csv", ".json", 
            ".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm"
        }
        
        self.bm25 = None
        self.bm25_corpus = [] # List of strings
        self.bm25_metadata = [] # List of dicts
        self._load_bm25_index()

    def _load_bm25_index(self):
        """BM25 corpusunu diskten yükler (varsa)."""
        corpus_path = self.db_dir / "bm25_corpus.json"
        if corpus_path.exists() and BM25Okapi:
            try:
                data = json.loads(corpus_path.read_text(encoding="utf-8"))
                self.bm25_corpus = data.get("corpus", [])
                self.bm25_metadata = data.get("metadata", [])
                if self.bm25_corpus:
                    tokenized_corpus = [doc.lower().split() for doc in self.bm25_corpus]
                    self.bm25 = BM25Okapi(tokenized_corpus)
            except Exception as e:
                log.error(f"BM25 index yükleme hatası: {e}")

    def _save_bm25_index(self):
        """BM25 corpusunu diske kaydeder."""
        corpus_path = self.db_dir / "bm25_corpus.json"
        try:
            corpus_path.write_text(json.dumps({
                "corpus": self.bm25_corpus,
                "metadata": self.bm25_metadata
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            log.error(f"BM25 index kaydetme hatası: {e}")
    def _chunk_text(self, text: str, chunk_size: int = 1500, overlap: int = 300) -> List[str]:
        """Metni parçalara (chunk) ayırır."""
        chunks = []
        if not text:
            return chunks
            
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            
            if end < text_len:
                # Break at a logical boundary if possible
                last_boundary = -1
                for char in ('\n', '.', '?', '!', ' ', ';'):
                    pos = text.rfind(char, start + chunk_size // 2, end)
                    if pos > last_boundary:
                        last_boundary = pos
                
                if last_boundary != -1:
                    end = last_boundary + 1
                    
            chunk_str = str(text[start:end]).strip()
            chunks.append(chunk_str)
            start = end - overlap
            if start >= text_len or end >= text_len:
                break
                
        return [c for c in chunks if c]

    def _load_pdf(self, path: Path) -> List[Tuple[str, Dict]]:
        """PDF dosyasını sayfa bazlı yükler."""
        if not pypdf:
            return [("[ERROR] pypdf yüklü değil.", {})]
        
        pages = []
        try:
            reader = pypdf.PdfReader(path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    pages.append((text, {"page": i + 1, "total_pages": len(reader.pages)}))
        except Exception as e:
            log.error(f"PDF yükleme hatası ({path}): {e}")
        return pages

    def _load_docx(self, path: Path) -> List[Tuple[str, Dict[str, Any]]]:
        """DOCX dosyasını yükler ve başlık bazlı bölümlere ayırır."""
        if not docx:
            return [("[ERROR] python-docx yüklü değil.", {})]
            
        sections: List[Tuple[str, Dict[str, Any]]] = []
        try:
            doc = docx.Document(path)
            current_section = "Başlangıç"
            current_text: List[str] = []
            
            for para in doc.paragraphs:
                # Başlık tespiti (Heading 1, Heading 2 vb.)
                style_name = para.style.name if para.style else ""
                if style_name.startswith('Heading'):
                    if current_text:
                        sections.append(("\n".join(current_text), {"section": str(current_section)}))
                        current_text = []
                    current_section = str(para.text).strip() or "Untitled Section"
                else:
                    current_text.append(str(para.text))
            
            if current_text:
                sections.append(("\n".join(current_text), {"section": current_section}))
            
            return sections if sections else [("", {})]
        except Exception as e:
            log.error(f"DOCX yükleme hatası ({path}): {e}")
            return []

    def _load_pptx(self, path: Path) -> List[Tuple[str, Dict]]:
        """PPTX dosyasını slide bazlı yükler."""
        if not pptx:
            return [("[ERROR] python-pptx yüklü değil.", {})]
            
        slides = []
        try:
            prs = pptx.Presentation(path)
            for i, slide in enumerate(prs.slides):
                slide_text = []
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        slide_text.append(shape.text)
                text = "\n".join(slide_text)
                if text.strip():
                    slides.append((text, {"slide": i + 1, "total_slides": len(prs.slides)}))
        except Exception as e:
            log.error(f"PPTX yükleme hatası ({path}): {e}")
        return slides

    def _load_excel(self, path: Path) -> List[Tuple[str, Dict]]:
        """XLSX dosyasını sheet bazlı yükler."""
        if not openpyxl:
            return [("[ERROR] openpyxl yüklü değil.", {})]
            
        sheets = []
        try:
            wb = openpyxl.load_workbook(path, data_only=True)
            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                rows = []
                for row in ws.iter_rows(values_only=True):
                    row_str = "\t".join([str(c) if c is not None else "" for c in row])
                    if row_str.strip():
                        rows.append(row_str)
                text = "\n".join(rows)
                if text.strip():
                    sheets.append((text, {"sheet_name": sheet_name}))
        except Exception as e:
            log.error(f"Excel yükleme hatası ({path}): {e}")
        return sheets

    def _load_html(self, path: Path) -> List[Tuple[str, Dict]]:
        """HTML dosyasını yükler."""
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            soup = BeautifulSoup(content, 'html.parser')
            # Gereksiz etiketleri temizle
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text(separator='\n')
            # Fazla boşlukları temizle
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            return [(text, {})]
        except Exception as e:
            log.error(f"HTML yükleme hatası ({path}): {e}")
            return []

    def index_workspace(self) -> int:
        """Workspace altındaki tüm desteklenen dosyaları bulur ve indeksler.
        Eskileri temizler ve baştan indeksler.
        
        Döndürdüğü değer işlenen doküman sayısıdır."""
        
        log.info("RAG indekslemesi başlatılıyor...")
        
        if self.client is None:
            log.error("chromadb import edilemediği için RAG indexlemesi atlanıyor.")
            return 0
        
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
        
        self.bm25_corpus = []
        self.bm25_metadata = []
        
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
                    ext = file_path.suffix.lower()
                    rel_path = str(file_path.relative_to(self.workspace_dir))
                    
                    # Dosya tipine göre içeriği yükle
                    file_contents = [] # List of (text, metadata)
                    
                    if ext in {".pdf"}:
                        file_contents = self._load_pdf(file_path)
                    elif ext in {".docx"}:
                        file_contents = self._load_docx(file_path)
                    elif ext in {".pptx"}:
                        file_contents = self._load_pptx(file_path)
                    elif ext in {".xlsx"}:
                        file_contents = self._load_excel(file_path)
                    elif ext in {".html", ".htm"}:
                        file_contents = self._load_html(file_path)
                    elif ext in {".md"}:
                        # Markdown için hiyerarşik yapı analizi
                        text = file_path.read_text(encoding="utf-8", errors="ignore")
                        lines = text.splitlines()
                        current_section = "Giriş"
                        current_content = []
                        
                        for line in lines:
                            if line.startswith("#"):
                                if current_content:
                                    file_contents.append(("\n".join(current_content), {"section": current_section}))
                                    current_content = []
                                current_section = line.lstrip("#").strip()
                            else:
                                current_content.append(line)
                        
                        if current_content:
                            file_contents.append(("\n".join(current_content), {"section": current_section}))
                    else:
                        # Düz metin tabanlı dosyalar (py, txt, json, csv)
                        text = file_path.read_text(encoding="utf-8", errors="ignore")
                        file_contents = [(text, {})]

                    for content_text, extra_meta in file_contents:
                        chunks = self._chunk_text(content_text)
                        
                        for i, chunk in enumerate(chunks):
                            chunk_text_str = str(chunk)
                            if not chunk_text_str.strip():
                                continue
                            
                            idx_str = str(i)
                            chunk_id = f"{rel_path}_{idx_str}_{len(docs)}"
                            
                            chunk_meta = {
                                "source": rel_path,
                                "file_type": ext,
                                "chunk_index": i
                            }
                            chunk_meta.update(extra_meta)
                            
                            docs.append(chunk_text_str)
                            metadatas.append(chunk_meta)
                            ids.append(chunk_id)
                            
                            # BM25 için sakla
                            self.bm25_corpus.append(chunk_text_str)
                            self.bm25_metadata.append(chunk_meta)
                            
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
                
        if self.bm25_corpus and BM25Okapi:
            try:
                # Tokenize
                tok_corpus = []
                for doc_c in self.bm25_corpus:
                    tok_corpus.append(str(doc_c).lower().split())
                    
                self.bm25 = BM25Okapi(tok_corpus)
                self._save_bm25_index()
            except Exception as e:
                log.error(f"BM25 build hatası: {e}")
                
        log.info(f"RAG indeksleme tamamlandı. {doc_count} dosya işlendi, {len(docs)} parça eklendi.")
        return doc_count

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Hybrid Search (Vektör + BM25)."""
        if not query.strip():
            return []
            
        # source_chunk -> result_dict
        combined_results: Dict[str, Dict[str, Any]] = {} 
        
        # 1. Vektör Araması (Semantic)
        if self.collection is not None:
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=top_k * 2 
                )
            
                doc_lists = results.get('documents')
                meta_lists = results.get('metadatas')
                dist_lists = results.get('distances')

                if doc_lists and doc_lists[0] and meta_lists and meta_lists[0] and dist_lists and dist_lists[0]:
                    for doc, meta, dist in zip(doc_lists[0], meta_lists[0], dist_lists[0]):
                        if not meta: continue
                        source = str(meta.get("source", "unknown"))
                        c_idx = meta.get("chunk_index", 0)
                        key = f"{source}_{c_idx}"
                        
                        combined_results[key] = {
                            "document": doc,
                            "source": source,
                            "section": meta.get("section", "N/A"),
                            "score": float(1.0 / (1.0 + dist)),
                            "type": "semantic"
                        }
            except Exception as e:
                log.error(f"Vektör arama hatası: {e}")

        # 2. Anahtar Kelime Araması (BM25)
        if self.bm25 and BM25Okapi:
            try:
                tokenized_query = query.lower().split()
                bm25_scores = self.bm25.get_scores(tokenized_query)
                # En iyi k sonucu al
                top_n_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k * 2]
                
                for idx in top_n_idx:
                    raw_score = float(bm25_scores[idx])
                    if raw_score <= 0: continue
                    
                    doc_text = self.bm25_corpus[idx]
                    meta_data = self.bm25_metadata[idx]
                    source_val = str(meta_data.get("source", "unknown"))
                    idx_val = meta_data.get("chunk_index", 0)
                    key = f"{source_val}_{idx_val}"
                    
                    boosted_score = raw_score * 0.1
                    
                    if key in combined_results:
                        combined_results[key]["score"] = float(combined_results[key]["score"]) + boosted_score
                        combined_results[key]["type"] = str(combined_results[key]["type"]) + "+keyword"
                    else:
                        combined_results[key] = {
                            "document": doc_text,
                            "source": source_val,
                            "section": meta_data.get("section", "N/A"),
                            "score": boosted_score,
                            "type": "keyword"
                        }
            except Exception as e:
                log.error(f"BM25 arama hatası: {e}")

        # Sırala ve en iyi top_k döndür
        all_results = list(combined_results.values())
        # Reranking: Skorları normalize et ve final sıralamayı yap
        sorted_res = sorted(all_results, key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return sorted_res[:top_k]
