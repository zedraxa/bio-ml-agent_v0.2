import logging
from typing import List, Dict, Any

log = logging.getLogger("anti_loop_governor")

class AntiLoopGovernor:
    """
    Anti-loop Governor: Tarayıcı adımları arasındaki kısır döngüleri tespit eder.
    Aynı aksiyonun aynı hedefle 3 kez denenmesi gibi durumları durdurur.
    """

    def __init__(self, repeat_threshold: int = 3):
        self.repeat_threshold = repeat_threshold
        self.action_history: List[Dict[str, Any]] = []

    def log_action(self, action_type: str, target: str, value: Any = None):
        """Her başarılı/başarısız aksiyonu kaydeder."""
        self.action_history.append({
            "type": action_type,
            "target": target,
            "value": value
        })

    def is_looping(self) -> bool:
        """Son adımlarda tekrar eden bir örüntü olup olmadığını kontrol eder."""
        if len(self.action_history) < self.repeat_threshold:
            return False

        last_actions = self.action_history[len(self.action_history) - self.repeat_threshold:]
        first_action = last_actions[0]

        # Basit Tekrarlı Döngü Kontrolü: Aynı tip ve target
        is_repeat = all(
            a["type"] == first_action["type"] and a["target"] == first_action["target"]
            for a in last_actions
        )

        if is_repeat:
            log.warning(f"🔄 Döngü tespit edildi! {first_action['type']} -> {first_action['target']}")
            return True

        return False

    def reset(self):
        self.action_history = []
