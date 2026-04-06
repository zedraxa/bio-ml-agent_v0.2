import requests
import string
import random
import time
from typing import Optional, Dict

class AgentEmailClient:
    """
    Mail.tm API'sini kullanarak Ajan için otonom tek kullanımlık/kalıcı email yönetimi sağlar.
    """
    API_URL = "https://api.mail.tm"

    def __init__(self, address: str = None, password: str = None):
        self.address = address
        self.password = password
        self.token = None

    @classmethod
    def get_agent_email(cls) -> "AgentEmailClient":
        """
        Yeni rastgele bir e-posta adresi ve şifre oluşturur.
        """
        try:
            res = requests.get(f"{cls.API_URL}/domains", timeout=10)
            res.raise_for_status()
            domains = res.json().get("hydra:member", [])
            if not domains:
                raise Exception("Kullanılabilir domain bulunamadı.")

            domain = domains[0]["domain"]

            # Rastgele e-posta oluştur
            random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
            address = f"bioagent_{random_str}@{domain}"
            password = ''.join(random.choices(string.ascii_letters + string.digits, k=16))

            # Hesabı kaydet
            create_res = requests.post(
                f"{cls.API_URL}/accounts",
                json={"address": address, "password": password},
                timeout=10
            )
            create_res.raise_for_status()

            client = cls(address, password)
            client._authenticate()
            return client

        except Exception as e:
            raise Exception(f"E-posta hesabı oluşturulurken hata: {e}")

    def _authenticate(self):
        """Hesap token'ını alır."""
        if not self.address or not self.password:
            raise ValueError("Kimlik bilgisi eksik.")

        res = requests.post(
            f"{self.API_URL}/token",
            json={"address": self.address, "password": self.password},
            timeout=10
        )
        res.raise_for_status()
        self.token = res.json().get("token")

    def get_messages(self, limit: int = 5) -> list:
        """Gelen kutusundaki son mesajları getirir."""
        if not self.token:
            self._authenticate()

        if not self.token:
            return []

        headers = {"Authorization": f"Bearer {self.token}"}
        res = requests.get(f"{self.API_URL}/messages", headers=headers, timeout=10)

        if res.status_code == 200:
            return res.json().get("hydra:member", [])[:limit]
        return []

    def get_message_content(self, message_id: str) -> Dict[str, str]:
        """Belirli bir mesajın tam içeriğini (Text ve HTML) okur."""
        if not self.token:
            self._authenticate()

        headers = {"Authorization": f"Bearer {self.token}"}
        res = requests.get(f"{self.API_URL}/messages/{message_id}", headers=headers, timeout=10)

        if res.status_code == 200:
            data = res.json()
            return {
                "subject": data.get("subject", ""),
                "from": data.get("from", {}).get("address", ""),
                "text": data.get("text", ""),
                "html": data.get("html", "")
            }
        return {}

    def wait_for_incoming_email(self, subject_contains: str = "", timeout_seconds: int = 60, poll_interval: int = 5) -> Optional[Dict[str, str]]:
        """
        Belirtilen konu başlığını içeren bir mail gelene kadar gelen kutusunu izler.
        Onay/Doğrulama mailleri için kullanılır.
        Bulduğu anda mesajın içeriğini (Subject, Text, HTML) döndürür.
        """
        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            messages = self.get_messages()
            for msg in messages:
                subj = msg.get("subject", "")
                if subject_contains.lower() in subj.lower():
                    # Mail bulundu, içeriğini çekelim
                    return self.get_message_content(msg["id"])

            time.sleep(poll_interval)

        return None

# Örnek kullanım (Ajanın BROWSER_AGENT esnasında nasıl kullanacağını görebilmesi için)
#
# from bio_ml_agent.utils.email_client import AgentEmailClient
# client = AgentEmailClient.get_agent_email()
# print("Email:", client.address)
# print("Password:", client.password)
#
# mail_content = client.wait_for_incoming_email(subject_contains="Verify", timeout_seconds=120)
# if mail_content:
#    # Regex ile doğrulama kodunu veya linkini ayıkla ve tarayıcı ile tıkla...
