# RAG System (Document Intelligence)

Bio-ML Agent uses a hybrid search infrastructure to understand project-specific documents.

## 📄 Supported Formats
- **Text**: `.txt`, `.py`, `.md`, `.js`, etc.
- **Office**: `.docx`, `.pptx`, `.xlsx`
- **Reports**: `.pdf`

## 🔎 Hybrid Search Algorithm
The key differentiator from other RAG solutions is the combination of two search methods:

1. **Vector Search (Semantic)**: Uses ChromaDB to search for the meaning of a sentence.
2. **Keyword Search (Lexical)**: Uses BM25 to search for technical terms (e.g., "helicase gene").

Both scores are merged in a **Reranker** layer (Reciprocal Rank Fusion) to surface the most relevant results to the agent.

## 🛠 Usage
You can instruct the agent to query documents with natural language:
- "According to the paper's conclusion section, what are the side effects?"
- "Find the DNA replication diagram from the Word document."

---

## 🇹🇷 Türkçe RAG Özeti (Turkish)

Bio-ML Agent, proje dokümanlarını anlamak için **hibrit arama** kullanır:

1. **Vektör Arama (Semantik)** — ChromaDB ile cümle anlamı aranır.
2. **Anahtar Kelime Arama (Leksikal)** — BM25 ile teknik terimler aranır.

İki skor **Reranker** katmanında birleştirilir. Ajana doğal dilde soru sorabilirsiniz:
- "Makalenin sonuç bölümüne göre yan etkiler nelerdir?"
- "DNA replikasyon şemasını Word dökümanından bul."
