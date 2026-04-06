import logging
import uuid
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime

log = logging.getLogger("kernel.message_bus")

@dataclass
class Message:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = "system"
    receiver: str = "broadcast"
    topic: str = "general"
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

class MessageBus:
    """
    MessageBus: Ajanlar arası asenkron ve pub/sub tabanlı iletişimi sağlar.
    Her mesajın takibini yapar ve 'Request-Response' patternini destekler.
    """

    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.history: List[Message] = []

    def subscribe(self, topic: str, callback: Callable):
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)
        log.info(f"📩 Subscribed to topic: {topic}")

    def publish(self, message: Message):
        self.history.append(message)
        log.debug(f"📤 Message Published: [{message.topic}] from {message.sender}")

        # Topic bazlı dağıtım
        if message.topic in self.subscribers:
            for callback in self.subscribers[message.topic]:
                try:
                    callback(message)
                except Exception as e:
                    log.error(f"Message callback error: {e}")

        # Broadcast (herkes duyabilir)
        if "broadcast" in self.subscribers and message.topic != "broadcast":
            for callback in self.subscribers["broadcast"]:
                callback(message)

    def request(self, sender: str, receiver: str, topic: str, data: Dict[str, Any]):
        """Senkron simülasyonu veya takip için request oluşturur."""
        msg = Message(sender=sender, receiver=receiver, topic=topic, payload=data)
        self.publish(msg)
        return msg.id
