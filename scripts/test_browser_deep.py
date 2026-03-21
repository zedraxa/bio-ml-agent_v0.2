import asyncio
import logging
from playwright.async_api import async_playwright
from bio_ml_agent.agents.browser.scout import BrowserScout
from bio_ml_agent.agents.browser.executor import BrowserExecutor

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("test.pubmed")

async def test_pubmed_search():
    """
    PubMed üzerinde gerçek bir arama ve veri toplama testi yapar.
    Sistemin 'layıkıyla' çalışıp çalışmadığını doğrular.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        log.info("🌐 Navigating to PubMed...")
        await page.goto("https://pubmed.ncbi.nlm.nih.gov/")
        
        context = {"page": page}
        
        # 1. Scout Aşaması
        scout = BrowserScout()
        log.info("🔭 Starting Scout...")
        await scout.perceive(context)
        result_scout = scout.summarize()
        log.info(f"✅ Scout Result: {result_scout.message}")
        
        # 2. Executor Aşaması (Arama yap)
        executor = BrowserExecutor()
        executor.perceive(context)
        
        log.info("🚀 Executing Search Action...")
        # Basit bir arama kutusuna 'CRISPR' yaz ve ENTER bas
        await page.fill("input[name='term']", "CRISPR")
        await page.keyboard.press("Enter")
        await page.wait_for_load_state("networkidle")
        
        log.info("🔭 Post-Search Scout...")
        await scout.perceive(context)
        
        await browser.close()
        log.info("🏁 PubMed Test Finished successfully.")

if __name__ == "__main__":
    asyncio.run(test_pubmed_search())
