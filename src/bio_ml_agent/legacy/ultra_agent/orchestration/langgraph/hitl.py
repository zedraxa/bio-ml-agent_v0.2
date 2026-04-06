import logging
from bio_ml_agent.ultra_agent.orchestration.langgraph.state import AgentState

log = logging.getLogger("bio_ml_agent")

def check_hitl_policy(state: AgentState, proposed_tool: str, proposed_args: str) -> AgentState:
    """
    S4-2: Security Policy Middleware for HITL 
    Belirli (riskli) toollar (örn. Autonomous API Registration) için HITL filtresi uygular.
    """

    # Antigravity policies dict
    GATED_TOOLS = {
        "BROWSER_ACTION": "Kullanıcı Onayı KESİNLİKLE Gereklidir (Web Navigasyonu).",
        "PYTHON": "Çok Yüksek CPU limitli işlemse Onay Gereklidir."
    }

    requires_approval = False
    if proposed_tool in GATED_TOOLS:
        log.warning(f"HITL KAPISI: {proposed_tool} aracı için onay bekleniyor! Neden: {GATED_TOOLS[proposed_tool]}")
        requires_approval = True

    state["requires_approval"] = requires_approval
    # Kullanıcıdan API üzerinden onay gelene kadar Graph burada "Interrupt" edilebilir.
    # Şimdilik approval_result boş bırakılıyor.
    state["approval_result"] = None

    return state
