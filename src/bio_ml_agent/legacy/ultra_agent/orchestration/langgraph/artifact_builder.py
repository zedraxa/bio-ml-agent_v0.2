import json
import logging
from pathlib import Path
from datetime import datetime

log = logging.getLogger("bio_ml_agent")

class ArtifactBuilder:
    """
    S4-3: Antigravity Standardına Uygun Artifact Çıktı Standardizasyonu.
    Her adımın sonucunda UI/UX da "markdown/json/image" formatında okunabilir, 
    kalıcı çıktılar bırakmak için kullanılır.
    """
    def __init__(self, workspace: Path):
        self.artifacts_dir = workspace / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def generate_artifact(self, name: str, content: str, artifact_type: str = "report") -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.{ 'json' if artifact_type == 'data' else 'md' }"

        filepath = self.artifacts_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            if artifact_type == "data":
                json.dump(content, f, indent=4)
            else:
                f.write(content)

        log.info(f"Artifact Standart Çıktı Üretildi: {filepath}")
        return filepath
