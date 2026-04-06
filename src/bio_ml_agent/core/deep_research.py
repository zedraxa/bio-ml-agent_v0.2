import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from bio_ml_agent.core.tools import web_search, web_open, write_file
from bio_ml_agent.llm_backend import auto_create_backend

log = logging.getLogger("bio_ml_agent")

class DeepResearchAgent:
    """
    Otonom Derin Araştırma Ajanı.
    Verilen konuyu web üzerinde derinlemesine araştırır ve kapsamlı bir rapor üretir.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash", workspace: Optional[Path] = None, project_name: str = "scratch_project"):
        self.model_name = model_name
        self.workspace = workspace or Path("workspace")
        self.project_name = project_name
        self.backend = auto_create_backend(model_name)
        self.collected_data = []

    def research(self, query: str, max_depth: int = 2) -> str:
        """
        Derin araştırma döngüsünü çalıştırır.
        """
        log.info(f"🔍 Derin Araştırma Başlatılıyor: {query}")

        # 1. Adım: Sorgu Ayrıştırma (Sub-queries)
        sub_queries = self._generate_sub_queries(query)
        log.info(f"📝 Alt Sorgular: {sub_queries}")

        # 2. Adım: Araştırma Döngüsü
        for sq in sub_queries:
            self._process_query(sq)

        # 3. Adım: Rapor Sentezleme
        report = self._synthesize_report(query)

        # 4. Adım: Raporu Kaydet
        report_filename = f"research/deep_report_{int(Path(__file__).stat().st_mtime)}.md"
        write_payload = f"path: {report_filename}\n---\n{report}"
        write_file(write_payload, self.workspace, project_name=self.project_name)

        return f"Araştırma tamamlandı. Rapor şuraya kaydedildi: {report_filename}\n\nÖzet:\n{report[:500]}..."

    def _generate_sub_queries(self, query: str) -> List[str]:
        prompt = f"""Kullanıcının araştırma konusu: "{query}"

Bu konuyu derinlemesine incelemek için 3-5 adet spesifik arama sorgusu oluştur. 
Sorgular biyomühendislik, tıp veya bilimsel verileri hedeflemeli.
SADECE sorguları içeren bir liste döndür (JSON formatında).

Örnek çıktı:
["query 1", "query 2", "query 3"]
"""
        try:
            res = self.backend.chat([{"role": "user", "content": prompt}])
            # JSON temizleme
            clean_res = res.strip()
            if "```json" in clean_res:
                clean_res = clean_res.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_res:
                clean_res = clean_res.split("```")[1].split("```")[0].strip()

            queries = json.loads(clean_res)
            return queries if isinstance(queries, list) else [query]
        except Exception as e:
            log.warning(f"Sorgu ayrıştırma hatası: {e}")
            return [query]

    def _process_query(self, query: str):
        """Tek bir sorgu için arama yapar ve içerik toplar."""
        try:
            search_results_raw = web_search(query)
            search_results = json.loads(search_results_raw)

            # Üstteki 2-3 sonucu oku
            for res in search_results[:3]:
                url = res.get("href")
                title = res.get("title")
                if not url: continue

                log.info(f"📖 Okunuyor: {title} ({url})")
                content = web_open(url)

                self.collected_data.append({
                    "query": query,
                    "title": title,
                    "url": url,
                    "content": content[:5000] # Bellek yönetimi için kısıtla
                })
        except Exception as e:
            log.warning(f"Sorgu işleme hatası ({query}): {e}")

    def _synthesize_report(self, original_query: str) -> str:
        """Toplanan tüm verilerden bir rapor oluşturur."""
        context = ""
        for i, item in enumerate(self.collected_data):
            context += f"\n--- KAYNAK {i+1} ({item['url']}) ---\n"
            context += f"Başlık: {item['title']}\n"
            context += f"İçerik: {item['content']}\n"

        prompt = f"""Aşağıdaki kaynaklardan gelen bilgileri kullanarak "{original_query}" konusu hakkında KAPSAMLI ve BİLİMSEL bir araştırma raporu yaz.

KURALLAR:
1. Dil: Türkçe olmalı.
2. Format: Markdown kullanılmalı.
3. Bölümler: Giriş, Metodoloji (nerelerden arandı), Detaylı Bulgular, Tartışma, Sonuç ve Kaynakça.
4. Bilimsel terimleri doğru kullan.
5. Görselleştirme önerileri ekle (eğer grafik çizilebilecek veri varsa).

TOPLANAN VERİLER:
{context[:20000]} # Limit check
"""
        try:
            report = self.backend.chat([{"role": "user", "content": prompt}])
            return report
        except Exception as e:
            log.error(f"Rapor sentezleme hatası: {e}")
            return f"Rapor oluşturulurken hata oluştu: {e}"
