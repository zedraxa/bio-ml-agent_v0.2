import logging
import pandas as pd
from typing import List, Dict, Any, Optional

log = logging.getLogger("document.table_extractor")

class TableExtractor:
    """
    TableExtractor: Yapısal veri ayıklama modülü.
    - Pandas ve Tabulate entegrasyonu ile Markdown verimliliği sağlar.
    """
    
    @staticmethod
    def extract_from_html(html_content: str) -> List[pd.DataFrame]:
        try:
            dfs = pd.read_html(html_content)
            log.info(f"📊 Extracted {len(dfs)} tables.")
            return dfs
        except Exception as e:
            log.debug(f"ℹ️ No tables in HTML: {e}")
            return []

    @staticmethod
    def to_markdown(df: pd.DataFrame) -> str:
        try:
            return df.to_markdown(index=False)
        except ImportError:
            log.error("❌ 'tabulate' library is missing. Install it to enable Markdown export.")
            return df.to_string()
        except Exception as e:
            log.error(f"❌ Markdown conversion error: {e}")
            return str(df)
