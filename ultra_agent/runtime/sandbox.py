import subprocess
import os
import resource
import logging
import sys
from pathlib import Path
from typing import Tuple, Dict

log = logging.getLogger("bio_ml_agent")

class SandboxRuntime:
    """
    OpenHands mimarisinden ilham alınarak yazılmış, python ortamlarını izole etmeye 
    çalışan v0 Sandbox denemesi.
    
    Not: Üretim (Production) aşamasında Container/gVisor tarzı araçlara geçilecektir.
    """
    
    def __init__(self, workspace: Path, work_dir: str = "/tmp", max_memory_mb: int = 2048):
        self.workspace = workspace
        self.work_dir = Path(work_dir)
        self.max_memory_mb = max_memory_mb
        
    def _set_resources(self):
        """Alt process için kaynak kısıtlamalarını ayarlar."""
        try:
            # CPU Limit (Soft 45 / Hard 60) saniye cinsinden
            resource.setrlimit(resource.RLIMIT_CPU, (45, 60))
            
            # Dinamik Bellek Limiti
            mem_limit = self.max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
            
            # File Descriptor Limiti
            resource.setrlimit(resource.RLIMIT_NOFILE, (128, 256))
        except (ValueError, OSError):
            pass

    def run_python_code(self, code: str, timeout: int = 60) -> Tuple[str, int]:
        """Verilen python kodunu kısıtlanmış kaynaklarla çalıştırır."""
        script_path = self.work_dir / "_sandbox_script.py"
        try:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)
                
            env = os.environ.copy()
            # Kritik çevresel değişkenleri maskele (Secret leakage protection)
            for k in list(env.keys()):
                if "API_KEY" in k or "TOKEN" in k or "PASSWORD" in k or "SECRET" in k:
                    env.pop(k, None)
                    
            # Güvenli Python izolasyon ortamı oluşturarak çalıştır
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                preexec_fn=self._set_resources
            )
            
            output = result.stdout + "\n" + result.stderr if result.stderr else result.stdout
            
            # Koca arrayler print edilirse terminali dondurmasın (Hard limit)
            if len(output) > 15000:
                output = output[:15000] + "\n\n... [SANDBOX: MAXIMUM OUTPUT TRUNCATED]"
                
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
