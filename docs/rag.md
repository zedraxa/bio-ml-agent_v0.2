# RAG Sistemi (Doküman Zekası)

Bio-ML Agent, projeye özel dökümanları anlamak için hibrit bir arama altyapısı kullanır.

## 📄 Desteklenen Formatlar
- **Metin:** `.txt`, `.py`, `.md`, `.js` vb.
- **Ofis:** `.docx`, `.pptx`, `.xlsx`
- **Raporlar:** `.pdf`

## 🔎 Hibrit Arama Algoritması
Sistemi diğer RAG çözümlerinden ayıran temel fark, iki arama yöntemini birleştirmesidir:

1.  **Vektör Arama (Semantic):** ChromaDB kullanarak cümlenin anlamını arar.
2.  **Anahtar Kelime Arama (Lexical):** BM25 kullanarak teknik terimleri (örn. "helikaz geni") arar.

Bu iki skor **Reranker** katmanında birleştirilerek en doğru sonuçlar ajana sunulur.

## 🛠 Kullanım
Ajana şu komutlarla doküman sormasını söyleyebilirsiniz:
- "Makalenin sonuç bölümüne göre yan etkiler nelerdir?"
- "DNA replikasyon şemasını Word dökümanından bul."
