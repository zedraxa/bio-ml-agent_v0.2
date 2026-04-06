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

    SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.csv', '.txt', '.html', '.md', '.xlsx', '.pptx'}

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
        elif ext == '.xlsx':
            return self._parse_xlsx(filepath)
        elif ext == '.pptx':
            return self._parse_pptx(filepath)
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

            # Metadata'yı kopyala ve zenginleştir
            chunk_metadata = base_metadata.copy()
            chunk_metadata["chunk_index"] = len(chunks)
            chunk_metadata["char_start"] = start
            chunk_metadata["char_end"] = end

            # Token limit/estimate (yaklaşık kelime sayısı * 1.3 formülü, LLM'lerde sık kullanılır)
            chunk_metadata["chunk_token_count"] = int(len(chunk_text.split()) * 1.3)

            # Mime type eşleştirme
            file_type = chunk_metadata.get("file_type", "")
            mime_map = {
                "pdf": "application/pdf",
                "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                "csv": "text/csv",
                "txt": "text/plain",
                "md": "text/markdown",
                "html": "text/html"
            }
            if file_type in mime_map:
                chunk_metadata["mime_type"] = mime_map[file_type]

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
            from docx import Document
            doc = Document(filepath)

            sections = []
            current_section = "General"

            for para in doc.paragraphs:
                cleaned_text = para.text.strip()
                if not cleaned_text:
                    continue
                # Başlıkları (Heading) tespit edip bölüm metaverisi yap
                if para.style.name.startswith("Heading"):
                    current_section = cleaned_text

                sections.append((current_section, cleaned_text))

            # Metinleri bölümler halinde gruplayarak chunk'la
            # Şimdilik en temizi tüm metni tekil gönderip section listesi tutmak ya da parça parça chunk'lamak
            # Daha stabil olması için klasik birleştirme yapıyoruz ancak section bilgisi ekliyoruz

            full_text = "\n".join([text for _, text in sections])
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

    def _parse_xlsx(self, filepath: Path) -> List[DocumentChunk]:
        chunks = []
        try:
            from openpyxl import load_workbook
            wb = load_workbook(filename=filepath, read_only=True, data_only=True)
            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                rows = list(ws.rows)
                if not rows:
                    continue

                # İlk satırı başlık olarak al
                headers = [str(cell.value) if cell.value is not None else f"Col_{i}" for i, cell in enumerate(rows[0])]
                sheet_texts = []

                for r_idx, row in enumerate(rows[1:], start=2):
                    row_data = [str(cell.value) if cell.value is not None else "" for cell in row]
                    if any(row_data):
                        row_dict = dict(zip(headers, row_data))
                        sheet_texts.append(f"Row {r_idx}: " + json.dumps(row_dict, ensure_ascii=False))

                if sheet_texts:
                    full_text = "\n".join(sheet_texts)
                    metadata = {
                        "source": filepath.name,
                        "file_type": "xlsx",
                        "sheet_name": sheet_name
                    }
                    chunks.extend(self._chunk_text(full_text, metadata))
        except ImportError:
            log.error(f"openpyxl modülü eksik. Lütfen 'pip install openpyxl' komutunu çalıştırın.")
        except Exception as e:
            log.error(f"XLSX Parse Hatası ({filepath.name}): {e}")
        return chunks

    def _parse_pptx(self, filepath: Path) -> List[DocumentChunk]:
        chunks = []
        try:
            from pptx import Presentation
            prs = Presentation(filepath)
            total_slides = len(prs.slides)

            for i, slide in enumerate(prs.slides):
                slide_texts = []
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text:
                        slide_texts.append(shape.text)
                    if shape.has_table:
                        for row in shape.table.rows:
                            row_data = [cell.text for cell in row.cells if cell.text]
                            if row_data:
                                slide_texts.append(" | ".join(row_data))

                text = "\n".join(slide_texts).strip()
                if text:
                    metadata = {
                        "source": filepath.name,
                        "file_type": "pptx",
                        "slide_number": i + 1,
                        "total_slides": total_slides
                    }
                    chunks.extend(self._chunk_text(text, metadata))
        except ImportError:
            log.error(f"python-pptx modülü eksik. Lütfen 'pip install python-pptx' komutunu çalıştırın.")
        except Exception as e:
            log.error(f"PPTX Parse Hatası ({filepath.name}): {e}")
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
