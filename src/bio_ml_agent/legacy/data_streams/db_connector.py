"""
db_connector.py
---------------
Bio-ML Agent projesinin "Active Learning" süreçleri için kurumsal veri tabanlarından 
(PostgreSQL, SQLite, MySQL) eğitim verisi çeken bağlayıcı donanım sınıfıdır.
"""

import os
import pandas as pd
from typing import Optional
from sqlalchemy import create_engine
import logging

log = logging.getLogger("bio_ml_agent")

class DBConnector:
    def __init__(self, db_url: Optional[str] = None):
        """
        :param db_url: SQLAlchemy connection string.
                       Örn: 'postgresql://postgres:password@localhost/health_db'
                       Varsayılan: 'sqlite:///workspace/active_learning.db'
        """
        self.db_url = db_url or os.getenv("DB_CONNECTION_STRING", "sqlite:///workspace/active_learning.db")
        self.engine = None
        self._connect()

    def _connect(self):
        try:
            self.engine = create_engine(self.db_url)
            log.info(f"🔌 DB Connector bağlandı: {self.db_url}")
        except Exception as e:
            log.error(f"❌ DB Connector bağlantı hatası: {e}")
            raise

    def fetch_data(self, query: str) -> pd.DataFrame:
        """
        Verilen SQL sorgusu ile veritabanından veri çeker ve DataFrame döner.
        :param query: Örn: 'SELECT * FROM patients WHERE is_analyzed=False'
        """
        try:
            return pd.read_sql_query(query, self.engine)
        except Exception as e:
            log.error(f"❌ Veri çekerken hata (Query: {query}): {e}")
            raise

    def get_new_data_since(self, table_name: str, timestamp_col: str, last_time: str) -> pd.DataFrame:
        """
        Belirli bir tarihten sonra eklenen taze kayıtları (yeni hasta verilerini vb.) otomatik çeker.
        Bu fonksiyon 'Retraining Pipeline' tarafından saat başı dinlenmek için çağrılabilir.
        """
        query = f"SELECT * FROM {table_name} WHERE {timestamp_col} > '{last_time}'"
        return self.fetch_data(query)

    def mark_as_analyzed(self, table_name: str, id_col: str, record_ids: list):
        """
        Model retrain edilirken kullanılan verilerin flag'ini günceller 
        (Böylece aynı veriyi tekrar eğitim setine almayız)
        """
        if not record_ids:
            return

        ids_str = ",".join(map(str, record_ids))
        query = f"UPDATE {table_name} SET is_analyzed = True WHERE {id_col} IN ({ids_str})"

        try:
            with self.engine.begin() as conn:
                conn.execute(query)
            log.info(f"✅ {len(record_ids)} adet kayıt analiz edilmiş olarak işaretlendi.")
        except Exception as e:
            log.error(f"❌ Güncelleme hatası: {e}")
            raise

# --- TEST ---
if __name__ == "__main__":
    db = DBConnector("sqlite:///workspace/test_db.sqlite")
    print("DB Connector başlatıldı.")
