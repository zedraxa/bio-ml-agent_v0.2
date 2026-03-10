import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import pypdf
import csv
import json
import logging

log = logging.getLogger("bio_ml_agent.rag")

class DocumentChunk:
    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text
        self.metadata = metadata

class FileParser:
    """Farklı formattaki dosyaları okuyup metadatalı chunk'lar (DocumentChunk) üreten Sınıf."""
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.csv', '.txt', '.html', '.md'}

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse_file(self, filepath: str | Path) -> List[DocumentChunk]:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Dosya bulunamadı: {filepath}")

        ext = filepath.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            log.warning(f"Desteklenmeyen dosya formatı: {ext}, düz metin (txt) olarak deneniyor.")
            ext = '.txt'

        if ext == '.pdf':
            return self._parse_pdf(filepath)
        elif ext == '.docx':
            return self._parse_docx(filepath)
        elif ext == '.csv':
            return self._parse_csv(filepath)
        elif ext in ('.txt', '.md', '.html'):
            return self._parse_text(filepath)
        else:
            return []

    def _chunk_text(self, text: str, base_metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Metni belirtilen boyut ve overlap'e göre böler."""
        if not text.strip():
            return []
            
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Metadata'yı kopyala ve chunk_index ekle
            chunk_metadata = base_metadata.copy()
            chunk_metadata["chunk_index"] = len(chunks)
            chunk_metadata["char_start"] = start
            chunk_metadata["char_end"] = end
            
            chunks.append(DocumentChunk(text=chunk_text, metadata=chunk_metadata))
            start += self.chunk_size - self.chunk_overlap
            
        return chunks

    def _parse_pdf(self, filepath: Path) -> List[DocumentChunk]:
        chunks = []
        try:
            reader = pypdf.PdfReader(filepath)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    metadata = {
                        "source": filepath.name,
                        "file_type": "pdf",
                        "page_number": i + 1,
                        "total_pages": len(reader.pages)
                    }
                    chunks.extend(self._chunk_text(text, metadata))
        except Exception as e:
            log.error(f"PDF Parse Hatası ({filepath.name}): {e}")
        return chunks

    def _parse_docx(self, filepath: Path) -> List[DocumentChunk]:
        chunks = []
        try:
            # Sadece yüklüyse import et
            from docx import Document
            doc = Document(filepath)
            full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            
            metadata = {
                "source": filepath.name,
                "file_type": "docx"
            }
            chunks.extend(self._chunk_text(full_text, metadata))
        except ImportError:
            log.error(f"python-docx modülü eksik. Lütfen 'pip install python-docx' komutunu çalıştırın.")
        except Exception as e:
            log.error(f"DOCX Parse Hatası ({filepath.name}): {e}")
        return chunks

    def _parse_csv(self, filepath: Path) -> List[DocumentChunk]:
        chunks = []
        try:
            with open(filepath, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader, None)
                
                row_texts = []
                for i, row in enumerate(reader):
                    if headers and len(headers) == len(row):
                        row_dict = dict(zip(headers, row))
                        row_texts.append(f"Row {i+1}: " + json.dumps(row_dict, ensure_ascii=False))
                    else:
                        row_texts.append(f"Row {i+1}: " + " | ".join(row))
                
                full_text = "\n".join(row_texts)
                metadata = {
                    "source": filepath.name,
                    "file_type": "csv"
                }
                chunks.extend(self._chunk_text(full_text, metadata))
        except Exception as e:
            log.error(f"CSV Parse Hatası ({filepath.name}): {e}")
        return chunks

    def _parse_text(self, filepath: Path) -> List[DocumentChunk]:
        chunks = []
        try:
            with open(filepath, mode='r', encoding='utf-8', errors='replace') as f:
                full_text = f.read()
            metadata = {
                "source": filepath.name,
                "file_type": filepath.suffix.lower().replace(".", "") or "txt"
            }
            chunks.extend(self._chunk_text(full_text, metadata))
        except Exception as e:
            log.error(f"Text Parse Hatası ({filepath.name}): {e}")
        return chunks
