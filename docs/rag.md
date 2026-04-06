# RAG System (Document Intelligence)

Bio-ML Agent uses a hybrid search infrastructure to understand project-specific documents.

## 📄 Supported Formats
- **Text**: `.txt`, `.py`, `.md`, `.js`, etc.
- **Office**: `.docx`, `.pptx`, `.xlsx`
- **Reports**: `.pdf`

## 🗄️ Vector Backend: Qdrant
**Qdrant** is the standardized backend for all semantic memory and RAG operations (see [ADR 0001](adr/0001-semantic-memory-backend-qdrant.md)).

Key Qdrant capabilities used:
- **TTL (Time-to-Live)**: Sensitive data expires automatically.
- **Project Scoping**: Memories are namespaced per project.
- **Provenance Tracking**: Every stored item records its source (file path or URL).

ChromaDB is retained as a legacy/fallback option in `src/bio_ml_agent/legacy/` but is no longer the primary backend.

## 🔎 Hybrid Search Algorithm
The system combines two complementary retrieval methods:

1. **Vector Search (Semantic)**: Uses Qdrant to search for the *meaning* of a sentence.
2. **Keyword Search (Lexical)**: Uses BM25 to search for exact technical terms (e.g., `"helicase gene"`, `"ATP synthase"`).

Both scores are fused using **Reciprocal Rank Fusion (RRF)** and optionally re-ranked by a cross-encoder to surface the most relevant chunks.

## 📥 Supported Ingestion Sources
- Individual files dropped into the workspace
- All `.docx`, `.xlsx`, `.pptx` office documents (structural extraction)
- PDF reports
- Web pages (via `WEB_OPEN` / `BROWSER_OPEN` tool results)

## 🛠 Usage
Instruct the agent to query documents with natural language:
- `"According to the paper's conclusion, what are the reported side effects?"`
- `"Find the DNA replication diagram from the Word document."`
- `"Summarize the method section of any uploaded PDFs."`

Or trigger workspace indexing manually:
```
>>> /ragindex
```

## ⚙️ Configuration (`config.yaml`)
```yaml
qdrant:
  host: localhost
  port: 6333
  collection_name: bio_ml_memory

memory:
  ttl_days: 30
  max_items: 10000
```

---

## 🇹🇷 Türkçe RAG Özeti (Turkish)

Bio-ML Agent, **Qdrant** vektör veritabanı üzerinde hibrit arama kullanır (ChromaDB eski/yedek olarak kalmaktadır).

**Arama Yöntemi:**
1. **Anlamsal (Vektör)** — Qdrant, cümle anlamı.
2. **Leksikal (BM25)** — Teknik terim araması.

İki skor **RRF** ile birleştirilir. Ajana doğal dilde soru sorabilirsiniz:
- `"Makalenin sonuç bölümüne göre yan etkiler nelerdir?"`
- `"DNA replikasyon şemasını Word dökümanından bul."`
