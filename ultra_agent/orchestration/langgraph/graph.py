import logging
from langgraph.graph import StateGraph, START, END

from ultra_agent.orchestration.langgraph.state import AgentState

log = logging.getLogger("bio_ml_agent")

def plan_step(state: AgentState):
    log.info("LangGraph [PLAN] Adımı Çalışıyor...")
    # Planlama mantığı (LLM çağrısı) vs.
    state_update = {
        "messages": [{"role": "assistant", "content": "[PLAN] Adımlar belirlendi."}],
        "current_step": "EXECUTE",
        "requires_approval": False # Kritik bir adım yoksa False
    }
    
    # S4-2: Eğer kritik bir tool çağrısı öngörülürse
    if "kritik_islem" in state.get('messages', [{}])[-1].get('content', '') and state.get("approval_mode") != 1:
        state_update["requires_approval"] = True
        log.info("HITL Kapısı: Onay bekleniyor.")

    return state_update

def execute_step(state: AgentState):
    log.info("LangGraph [EXECUTE] Adımı Çalışıyor...")
    
    if state.get("requires_approval") and not state.get("approval_result") and state.get("approval_mode") != 1:
        # Onay verilmediyse dur.
        raise Exception("İşlem onaylanmadı veya henüz izin verilmedi.")
        
    state_update = {
        "messages": [{"role": "assistant", "content": "[EXECUTE] Tool çağrıldı."}],
        "current_step": "VERIFY"
    }
    return state_update

def verify_step(state: AgentState):
    log.info("LangGraph [VERIFY] Adımı Çalışıyor...")
    # S4-4: Comment-to-Iterate Kontrolü
    if state.get("feedback"):
        log.warning(f"Kullanıcı revizyon istedi: {state['feedback']}. PLAN'a geri dönülüyor.")
        return {"current_step": "PLAN", "feedback": None, "messages": [{"role": "user", "content": f"Revizyon Görevi: {state['feedback']}"}]}
    
    return {"current_step": "ARTIFACT"}

def artifact_step(state: AgentState):
    log.info("LangGraph [ARTIFACT] Adımı Çalışıyor...")
    # S4-3: Sonuçları / Artifact'leri üret
    return {"messages": [{"role": "assistant", "content": "[ARTIFACT] Dosyalar oluşturuldu."}], "current_step": "END"}


def build_graph():
    """
    Topolojiyi (Plan -> Tool -> Verify -> Artifact) kurgular. S4-1.
    """
    builder = StateGraph(AgentState)
    
    # Nodları ekle
    builder.add_node("plan", plan_step)
    builder.add_node("execute", execute_step)
    builder.add_node("verify", verify_step)
    builder.add_node("artifact", artifact_step)
    
    # Kenarları (Edge) tanımla
    builder.add_edge(START, "plan")
    
    # Router mantığı (conditional edges)
    # Basitçe sırayla gidiyoruz, error/feedback olursa verify'da hallediliyor
    builder.add_edge("plan", "execute")
    builder.add_edge("execute", "verify")
    
    # Verify'dan Condition üzerinden Plan'a veya Artifact'a dallanma
    def verify_router(state: AgentState):
        if state.get("current_step") == "PLAN":
            return "plan"
        return "artifact"
        
    builder.add_conditional_edges("verify", verify_router)
    builder.add_edge("artifact", END)
    
    graph = builder.compile()
    return graph

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    g = build_graph()
    log.info("Graph topology başarıyla derlendi.")
