from typing import Annotated, Dict, Any, List, Optional
from typing_extensions import TypedDict
import operator

class AgentState(TypedDict):
    """
    S4-1: LangGraph için temel durum sınıfı (Graph State Topology).
    Plan -> Tool -> Verify -> Artifact döngüsündeki durumu tutar.
    
    `messages` listesi, her yeni gönderilen içeriği (append) alır.
    """
    # Mesajların geçmişi ve gidişatı
    messages: Annotated[List[Dict[str, Any]], operator.add]

    # Mevcut adım state (PLAN, EXECUTE, VERIFY vb.)
    current_step: str

    # Kullanıcıdan human-in-the-loop için onay durumu (S4-2)
    requires_approval: bool
    approval_result: Optional[str]

    # Hata yönetimi
    error_counter: int

    # Comment-to-Iterate geribildirimi (S4-4)
    feedback: Optional[str]

    # Onay Modu (1=Full Auto, 2=Interval, vb.)
    approval_mode: int
