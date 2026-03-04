import subprocess
import os
import resource
import logging
from pathlib import Path
from typing import Tuple, Dict

log = logging.getLogger("bio_ml_agent")

class SandboxRuntime:
    """
    OpenHands mimarisinden ilham alınarak yazılmış, python ortamlarını izole etmeye 
    çalışan v0 Sandbox denemesi.
    
    Not: Üretim (Production) aşamasında Container/gVisor tarzı araçlara geçilecektir.
    """
    
    def __init__(self, workspace: Path, work_dir: str = "/tmp"):
        self.workspace = workspace
        self.work_dir = Path(work_dir)
        
    def _set_resources(self):
        """Alt process için kaynak kısıtlamalarını ayarlar."""
        # CPU Limit (Soft / Hard) saniye cinsinden
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (30, 60))
            
            # Bellek Limiti 2 GB (ML kütüphaneleri için genişletildi)
            mem_limit = 2048 * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
            
            # File Descriptor Limiti
            resource.setrlimit(resource.RLIMIT_NOFILE, (128, 256))
        except (ValueError, OSError) as e:
            # İşletim sistemi limitleri değiştirmeye izin vermezse (örn. Mac OS veya root değilsek)
            pass

    def run_python_code(self, code: str, timeout: int = 25) -> Tuple[str, int]:
        """Verilen python kodunu kısıtlanmış kaynaklarla çalıştırır."""
        
        # Geçici python dosyası oluştur
        script_path = self.work_dir / "_sandbox_script.py"
        try:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)
                
            env = os.environ.copy()
            # Kritik çevresel değişkenleri maskele (Secret leakage protection)
            env.pop("OPENAI_API_KEY", None)
            env.pop("ANTHROPIC_API_KEY", None)
            env.pop("REDIS_HOST", None)
            
            # Güvenli Python izolasyon ortamı oluşturarak çalıştır
            result = subprocess.run(
                ["python3", str(script_path)],
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                preexec_fn=self._set_resources
            )
            
            output = result.stdout + "\n" + result.stderr if result.stderr else result.stdout
            return output.strip(), result.returncode
            
        except subprocess.TimeoutExpired:
            log.warning("Sandbox Runtime Timeout: Kod %ds'de bitmedi.", timeout)
            from exceptions import ToolTimeoutError
            raise ToolTimeoutError("PYTHON", timeout)
            
        except Exception as e:
            log.error(f"Sandbox Runtime Error: {e}")
            return f"❌ SANBOX_ERROR: Beklenmedik izolasyon hatası: {str(e)}", 1
            
        finally:
            if script_path.exists():
                script_path.unlink()

        return "❌ SANBOX_ERROR: Bilinmeyen hata yolu", 1
