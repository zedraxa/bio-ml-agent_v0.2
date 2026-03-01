"""
kafka_redis_consumer.py
-----------------------
Gerçek zamanlı akan (Stream) EKG, EEG cihazları veya Webhook JSON 
verilerini dinleyen (Consumer) asenkron kuyruk dinleyicisi. 
Projede Apache Kafka ağırlığı yerine hazır Redis imajı olduğu için 
Redis Streams kullanılarak yazılmıştır.
"""

import os
import json
import time
import redis
import logging
from typing import Callable, Any

log = logging.getLogger("bio_ml_agent")

class StreamConsumer:
    def __init__(self, stream_name: str, group_name: str, consumer_name: str):
        """
        Redis tabanlı hafifletilmiş Stream/Kafka kuyruğunu dinler.
        """
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        self.stream_name = stream_name
        self.group_name = group_name
        self.consumer_name = consumer_name
        
        try:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            self.redis_client.ping()
            # Tüketici grubunu başlat (Stream yoksa oluşturur)
            try:
                self.redis_client.xgroup_create(self.stream_name, self.group_name, id='0', mkstream=True)
                log.info(f"📡 Redis Consumer Group kuruldu: {self.group_name}")
            except redis.exceptions.ResponseError as e:
                if "BUSYGROUP" in str(e):
                    pass # Grup zaten var, sorun yok
                else:
                    raise e
                    
        except Exception as e:
            log.error(f"❌ Redis Stream bağlantı hatası: {e}")
            raise

    def publish_test_message(self, message: dict):
        """Streaming hattını test etmek veya Producer gibi davranıp veri yollamak için kullanılır."""
        payload = {k: str(v) for k, v in message.items()} # Redis dict value'ları string bekler
        self.redis_client.xadd(self.stream_name, payload)
        log.info(f"📤 Stream'e veri basıldı: {self.stream_name}")

    def listen(self, batch_size: int, callback: Callable[[list[dict]], Any], poll_timeout_ms: int = 5000):
        """
        Kuyruğu sonsuz döngüde dinler. Veri geldiğinde 'batch_size' kadarını toplayıp
        callback fonksiyonuna (Örn: Model Retraining servisine) atar.
        """
        log.info(f"🎧 Stream dinleniyor: {self.stream_name} (Group: {self.group_name})")
        
        while True:
            try:
                # Bloklayıcı okuma (timeout süresince bekler)
                messages = self.redis_client.xreadgroup(
                    groupname=self.group_name, 
                    consumername=self.consumer_name, 
                    streams={self.stream_name: '>'}, 
                    count=batch_size, 
                    block=poll_timeout_ms
                )
                
                if messages:
                    stream, msg_list = messages[0]
                    parsed_messages = []
                    msg_ids = []
                    
                    for msg_id, payload in msg_list:
                        parsed_messages.append(payload)
                        msg_ids.append(msg_id)
                        
                    # 1. Callback çalıştır (Örn: 50 tansiyon verisi geldi, modeli tetikle)
                    log.info(f"📥 {len(parsed_messages)} yeni veri akışı alındı.")
                    callback(parsed_messages)
                    
                    # 2. İşlenen mesajları onaylanmış (ACK) işaretle ki bir daha çekilmesin
                    self.redis_client.xack(self.stream_name, self.group_name, *msg_ids)
            
            except Exception as e:
                log.error(f"❌ Akış dinleme hatası: {e}")
                time.sleep(5) # Hata olursa kısa süre bekle ve tekrar dene

# --- TEST ---
def _mock_callback(batch):
    print(f"Mock Callback tetiklendi! Gelen veri boyutu: {len(batch)}")
    for data in batch:
        print(f"  Veri: {data}")

if __name__ == "__main__":
    consumer = StreamConsumer("sensor_stream", "active_learning_group", "worker_1")
    # Örnek test verisi bas
    consumer.publish_test_message({"heart_rate": 84, "timestamp": "2026-03-01T20:20:00Z"})
    print("Test verisi basıldı. Dinleme başlatılıyor (1 mesaj okuyup çıkacak)...")
    # Gerçek scriptte bu sonsuz çalışır, demo için:
    messages = consumer.redis_client.xreadgroup(consumer.group_name, consumer.consumer_name, {consumer.stream_name: '>'}, count=1, block=2000)
    print("Çekilen test:", messages)
