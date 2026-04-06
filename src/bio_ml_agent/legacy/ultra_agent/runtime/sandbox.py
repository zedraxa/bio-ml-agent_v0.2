import subprocess
import os
import resource
import logging
import sys
import ast
from pathlib import Path
from typing import Tuple, Dict

log = logging.getLogger("bio_ml_agent")

class SecurityError(Exception):
    pass

class CodeValidator(ast.NodeVisitor):
    """
    Python betiğini çalıştırmadan önce AST seviyesinde zararlı kod denetimi (Allowlist) yapar.
    """
    # Veri bilimi ve genel amaçlı, sisteme zarar veremeyecek kütüphaneler
    ALLOWED_MODULES = {
        'math', 'datetime', 'json', 're', 'collections', 'itertools',
        'typing', 'random', 'hashlib', 'time', 'uuid',
        'numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib', 'seaborn',
        'statsmodels', 'xgboost', 'lightgbm'
    }

    # Yasaklı (Blacklist) fonksiyon çağrıları
    BLOCKED_CALLS = {'eval', 'exec', 'open', 'compile', '__import__', 'globals', 'locals', 'vars'}

    # Dunder metotlar (Reflection / Jailbreak engelleyici)
    BLOCKED_ATTRS = {'__class__', '__bases__', '__subclasses__', '__builtins__', '__globals__', '__getattribute__'}

    def __init__(self):
        self.errors = []

    def visit_Import(self, node):
        for alias in node.names:
            base_module = alias.name.split('.')[0]
            if base_module not in self.ALLOWED_MODULES:
                self.errors.append(f"Güvenlik İhlali: İzin verilmeyen modül yüklendi (import) -> '{alias.name}'")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            base_module = node.module.split('.')[0]
            if base_module not in self.ALLOWED_MODULES:
                self.errors.append(f"Güvenlik İhlali: İzin verilmeyen modülden aktarım yapıldı -> '{node.module}'")
        self.generic_visit(node)

    def visit_Call(self, node):
        if hasattr(node.func, 'id'):
            if node.func.id in self.BLOCKED_CALLS:
                self.errors.append(f"Güvenlik İhlali: Yasaklı fonksiyon çağrısı -> '{node.func.id}'")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if node.attr in self.BLOCKED_ATTRS:
             self.errors.append(f"Güvenlik İhlali: Yasaklı nitelik (attribute) erişimi -> '{node.attr}'")
        self.generic_visit(node)

    def visit_Name(self, node):
        if node.id in self.BLOCKED_ATTRS:
            self.errors.append(f"Güvenlik İhlali: Yasaklı sistem değişkeni erişimi -> '{node.id}'")
        self.generic_visit(node)

def validate_code(code: str):
    """Zararlı scriptleri parse aşamasında yakalatan süzgeç."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SecurityError(f"Sözdizimi Hatası (SyntaxError): {e}")

    validator = CodeValidator()
    validator.visit(tree)

    if validator.errors:
        raise SecurityError("\n".join(validator.errors))

class SandboxRuntime:
    """
    OpenHands mimarisinden ilham alınarak yazılmış, python ortamlarını izole etmeye 
    çalışan v0 Sandbox denemesi.
    AST Tabanlı Güvenlik Modeli ile güçlendirilmiştir.
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
        """Verilen python kodunu kısıtlanmış kaynaklarla ve AST validasyonundan geçirerek çalıştırır."""

        # 1. AST Statik Kod Güvenlik Denetimi
        try:
            validate_code(code)
        except SecurityError as e:
            log.warning("Sandbox Engelledi: Zararlı/Yasaklı kod tespit edildi.")
            return f"❌ SANBOX_SECURITY_ERROR:\n{e}", 1

        # 2. Subprocess İzolasyonu
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
            from bio_ml_agent.exceptions import ToolTimeoutError
            raise ToolTimeoutError("PYTHON", timeout)

        except Exception as e:
            log.error(f"Sandbox Runtime Error: {e}")
            return f"❌ SANBOX_ERROR: Beklenmedik izolasyon hatası: {str(e)}", 1

        finally:
            if script_path.exists():
                script_path.unlink()

        return "❌ SANBOX_ERROR: Bilinmeyen hata yolu", 1
