from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

class BaseWorkspace(ABC):
    """Workspace için temel soyutlama katmanı."""
    
    @abstractmethod
    def read_file(self, path: str) -> str:
        """Bağıl yoldaki dosyayı okur."""
        pass
        
    @abstractmethod
    def write_file(self, path: str, content: str) -> None:
        """Bağıl yola dosya yazar."""
        pass
        
    @abstractmethod
    def list_files(self, prefix: str = "") -> List[str]:
        """Workspace altındaki dosyaları listeler."""
        pass

class LocalWorkspace(BaseWorkspace):
    """Yerel diskteki workspace."""
    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        
    def read_file(self, path: str) -> str:
        file_path = self.root_dir / path
        if not file_path.exists():
            raise FileNotFoundError(f"{path} bulunamadı.")
        return file_path.read_text(encoding="utf-8")
        
    def write_file(self, path: str, content: str) -> None:
        file_path = self.root_dir / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        
    def list_files(self, prefix: str = "") -> List[str]:
        target_dir = self.root_dir / prefix
        if not target_dir.exists():
            return []
        return [str(p.relative_to(self.root_dir)) for p in target_dir.rglob("*") if p.is_file()]

class SyncedCloudWorkspace(BaseWorkspace):
    """Uzaktaki depolama ile senkronize workspace."""
    def read_file(self, path: str) -> str:
        raise NotImplementedError("Henüz implement edilmedi")
        
    def write_file(self, path: str, content: str) -> None:
        raise NotImplementedError()
        
    def list_files(self, prefix: str = "") -> List[str]:
        raise NotImplementedError()

class EphemeralRemoteWorkspace(BaseWorkspace):
    """Geçici, işlem bitince yok olan workspace."""
    def read_file(self, path: str) -> str:
        raise NotImplementedError("Henüz implement edilmedi")
        
    def write_file(self, path: str, content: str) -> None:
        raise NotImplementedError()
        
    def list_files(self, prefix: str = "") -> List[str]:
        raise NotImplementedError()

class ReadOnlyReferenceWorkspace(BaseWorkspace):
    """RAG vs için sadece okunabilir referans workspace."""
    def read_file(self, path: str) -> str:
        raise NotImplementedError("Henüz implement edilmedi")
        
    def write_file(self, path: str, content: str) -> None:
        raise TypeError("ReadOnlyReferenceWorkspace salt okunurdur, yazılamaz.")
        
    def list_files(self, prefix: str = "") -> List[str]:
        raise NotImplementedError()
