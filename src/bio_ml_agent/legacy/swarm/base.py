from typing import List, Dict, Any, Optional

class SwarmContext:
    """Paylaşılan Swarm Belleği ve Context'i"""
    def __init__(self, workspace_path: str, model: str):
        self.workspace = workspace_path
        self.model = model
        self.shared_memory: Dict[str, Any] = {}
        self.history: List[Dict[str, str]] = []

class BaseAgent:
    """
    Sub-Agent'lar için temel sınıf.
    Her ajan kendi LLM backend'i ile haberleşebilir,
    genel Swarm Context'ine yazıp okuyabilir.
    """

    def __init__(self, name: str, role: str, context: SwarmContext):
        self.name = name
        self.role = role
        self.context = context

    def get_system_prompt(self) -> str:
        """Ajanın spesifik uzmanlık promptu"""
        raise NotImplementedError

    def execute(self, task_prompt: str = "", error_history: str = "") -> str:
        """Ajanın ana çalışma döngüsü"""
        raise NotImplementedError
