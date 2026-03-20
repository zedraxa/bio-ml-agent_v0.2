# multi_agent.py
# ═══════════════════════════════════════════════════════════
#  Bio-ML Agent — Multi-Agent Collaboration Module
#
#  Ana orchestrator agent'a (agent.py) yardımcı olacak alt
#  uzman ajanları barındırır. Her ajanın kendi özel sistem
#  prompt'u vardır ancak aynı workspace üzerinde ve kısıtlı
#  step sayısı ile çalışırlar.
# ═══════════════════════════════════════════════════════════

from typing import Dict, Any, Optional
from pathlib import Path
import os

# Agent core importları (Döngüsel importu önlemek için fonksiyon içinde kullanılacak)
# from agent import _cfg, llm_chat, extract_tool, run_python, run_bash, read_file, write_file

# ─────────────────────────────────────────────
#  Sub-Agent Prompt Tanımları
# ─────────────────────────────────────────────

DATA_ENGINEER_PROMPT = """You are the Data Engineer Sub-Agent.
Your objective is to clean, preprocess, and engineer features for ML datasets.

HARD RULES:
- Use ONLY the tool protocol (<PYTHON>, <BASH>, <READ_FILE>, <WRITE_FILE>).
- BASH commands run from the project workspace directory.
- Outputs must be reproducible saved files (e.g. data/processed/cleaned.csv).
- Do NOT train models. Your only job is data preparation.

When finished, respond with a summary of the changes you made to the data and the path to the saved file.
"""

ML_ENGINEER_PROMPT = """You are the Machine Learning Engineer Sub-Agent.
Your objective is to train, optimize, and evaluate ML models on prepared datasets.

HARD RULES:
- Use ONLY the tool protocol (<PYTHON>, <BASH>, <READ_FILE>, <WRITE_FILE>).
- You can use `from utils.model_compare import compare_models` and `from utils.hyperparameter_optimizer import optimize_model`.
- Save all results to the expected output directories (results/).
- Do NOT do extensive raw data cleaning; expect clean data.

When finished, respond with the performance metrics and the path to the best saved model (.pkl).
"""

REPORT_WRITER_PROMPT = """You are the Report Writer Sub-Agent.
Your objective is to synthesize data findings, model metrics, and plots into a markdown report.

HARD RULES:
- Use ONLY the tool protocol (<PYTHON>, <BASH>, <READ_FILE>, <WRITE_FILE>).
- Read the results JSON files and plot images.
- Generate a comprehensive, well-structured `report.md`.
- Be professional.

When finished, respond with a summary of the report and its saved path.
"""

# ─────────────────────────────────────────────
#  Sub-Agent Çalıştırma Motoru
# ─────────────────────────────────────────────

def _run_subagent(role_name: str, sys_prompt: str, task: str, max_steps: int = 5) -> str:
    """Belirtilen role sahip alt-ajanı çalıştırır ve nihai metin sonucunu döndürür."""
    # İçeriden import ediyoruz ki agent.py ile döngüsel bağımlılık olmasın.
    from legacy.agent import (
        _cfg, llm_chat, extract_tool, run_python, run_bash, read_file, write_file
    )
    import logging
    
    log = logging.getLogger("bio_ml_agent")
    cfg = _cfg()
    
    if not cfg:
        return f"Hata: Global konfigurasyon bulunamadı. Sub-agent {role_name} baslatilamiyor."

    project_name = os.environ.get("AGENT_PROJECT", "multi_agent_project")
    
    # cfg.workspace genelde Path objesidir, ama Config kullanildiginda base_dir icinde olabilir.
    try:
        base_ws = Path(getattr(cfg.workspace, "base_dir", str(cfg.workspace)))
    except Exception:
        base_ws = Path("workspace")

    workspace = base_ws / project_name
    workspace.mkdir(parents=True, exist_ok=True)

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"TASK FOR {role_name}:\n{task}\n\nDo not ask questions, just execute and report your final output."}
    ]

    log.info("🤖 Sub-Agent [%s] başlatıldı. Limit: %d adım", role_name, max_steps)
    print(f"\\n[{role_name} 🤖 çalışıyor...]")

    final_result = ""

    model = getattr(cfg.agent, "model", "qwen2.5:7b-instruct") if hasattr(cfg, "agent") else getattr(cfg, "model", "qwen2.5:7b-instruct")

    for step in range(max_steps):
        try:
            assistant = llm_chat(model, messages)
        except Exception as e:
            return f"{role_name} LLM Error: {e}"

        tool, payload, outside, _attrs = extract_tool(assistant)

        if tool is None:
            final_result = outside or assistant
            break

        # Sadece izin verilen core tool'ları çalıştır
        try:
            timeout = getattr(cfg, 'timeout', 180)
            if tool == "PYTHON":
                out = run_python(payload, workspace, timeout_s=timeout)
            elif tool == "BASH":
                out = run_bash(payload, workspace, timeout_s=timeout)
            elif tool == "READ_FILE":
                out = read_file(payload, workspace)
            elif tool == "WRITE_FILE":
                out = write_file(payload, workspace)
            else:
                out = f"Error: Tool {tool} is not allowed for sub-agents."
        except Exception as e:
            out = f"Tool Execution Error: {e}"

        messages.append({"role": "assistant", "content": assistant})
        messages.append({
            "role": "user",
            "content": f"TOOL_OUTPUT ({tool}):\\n{out}\\n\\nContinue. If the task is fully complete, answer normally without any tool tags."
        })
        
        final_result = outside or "Çalışma devam ediyor..."

    log.info("✅ Sub-Agent [%s] sonlandı.", role_name)
    print(f"[{role_name} ✅ tamamlandı]\\n")
    return final_result


# ─────────────────────────────────────────────
#  Public API (Orchestrator'ın çağıracağı fonk.)
# ─────────────────────────────────────────────

def ask_data_engineer(task: str) -> str:
    """Data Engineer alt-ajanına görev verir."""
    return _run_subagent("DATA_ENGINEER", DATA_ENGINEER_PROMPT, task, max_steps=8)

def ask_ml_engineer(task: str) -> str:
    """ML Engineer alt-ajanına görev verir."""
    return _run_subagent("ML_ENGINEER", ML_ENGINEER_PROMPT, task, max_steps=10)

def ask_report_writer(task: str) -> str:
    """Report Writer alt-ajanına görev verir."""
    return _run_subagent("REPORT_WRITER", REPORT_WRITER_PROMPT, task, max_steps=5)

